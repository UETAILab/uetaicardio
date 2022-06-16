import argparse
import json
import time

import cv2
import torch
from easydict import EasyDict
import numpy as np

from inference.inferencer import generate_reader, generate_classifier, generate_detector, generate_segmentator, \
    generate_analyzers, generate_coefficients_estimator


def get_parser():
    parser = EasyDict()
    parser["classifier_module_name"] = "classification"
    parser["chamber_classifier"] = "MobilenetV2"
    parser["classifier_weights"] = "/data.local/data/models/chamber_classification/mobilenetv2_0049_0.9507_best.pth"
    parser["classifier_batch_size"] = 32

    parser["detector_model_def"] = "PyTorch_YOLOv3/config/yolov3-custom.cfg"
    parser["detector_model_weights"] = "/data.local/data/models/YOLO-2C/0.7435_yolov3_ckpt_75.pth"
    parser["detector_batch_size"] = 2

    parser["segmentator_module_name"] = "segmentation"
    parser["full_frame_segmentator"] = "AuxUNet"
    parser[
        "full_frame_weights"] = "/data.local/data/models/segmentation/full2C/full_aux_giangunet_invhd_0009_0.8604_best.pth"
    parser["cropped_frame_segmentator"] = "Resnet101DeeplabV3"
    parser[
        "cropped_frame_weights"] = "/data.local/data/models/segmentation/cropped2C/cropped_aux_hd_resnet101_b8_vloss_0079_0.8707_best.pth"

    parser["device"] = "cuda:0"
    return parser


class ModelRuntime():
    def __init__(self, ):
        args = get_parser()

        args.classifier_weights = "/deploy_weight/mobilenetv2.pth"
        args.detector_model_weights = "/deploy_weight/yolov3.pth"
        args.full_frame_weights = "/deploy_weight/full_aux_giangunet_invhd.pth"
        args.cropped_frame_weights = "/deploy_weight/cropped_aux_hd_resnet101.pth"

        # args.classifier_weights = "D:/UETAILAB/deploy_weight/mobilenetv2.pth"
        # args.detector_model_weights = "D:/UETAILAB/deploy_weight/yolov3.pth"
        # args.full_frame_weights = "D:/UETAILAB/deploy_weight/full_aux_giangunet_invhd.pth"
        # args.cropped_frame_weights = "D:/UETAILAB/deploy_weight/cropped_aux_hd_resnet101.pth"

        self.reader = generate_reader(args)
        self.classifier = generate_classifier(args)
        self.detector = generate_detector(args)
        self.segmentator = generate_segmentator(args)

        (self.coarse_analyzer,
         self.fine_analyzer,
         self.pivot_extractor) = generate_analyzers(args)
        self.coefficients_estimator = generate_coefficients_estimator(args)

    def __read_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        frames = []
        while ret:
            frames.append(frame[None, ...])
            ret, frame = cap.read()
        return np.concatenate(frames)

    def __extend_contours(self, sample_frame, contours, pivot_points):
        extended_contours = []
        for idx, contour in enumerate(contours):
            left_point, right_point = np.array(pivot_points[idx][0]), np.array(pivot_points[idx][-1])
            points = np.array(pivot_points[idx])
            center_point = np.mean(points, axis=0)
            image = np.zeros_like(sample_frame)
            eroded = cv2.dilate(cv2.drawContours(image, [contour], 0, (255, 255, 255), thickness=-1),
                                np.array(([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]])), iterations=15)
            eroded = cv2.cvtColor(eroded, cv2.COLOR_BGR2GRAY)
            eroded_cnts, _ = cv2.findContours(np.uint8(eroded), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            eroded_cnts = eroded_cnts[0]
            is_above_left = np.cross(np.array(eroded_cnts) - left_point, center_point - left_point) > 0
            is_above_right = np.cross(np.array(eroded_cnts) - center_point, right_point - center_point) > 0
            is_above = np.bitwise_or(is_above_right, is_above_left)
            extended_contours.append(eroded_cnts[is_above])
        return extended_contours

    def predict(self, file_path, metadata, scale=1):
        frames = self.__read_video(file_path)

        tik = time.time()
        chamber = self.classifier.run(frames)
        print("classify", time.time() - tik)

        tik = time.time()
        global_bbox, bboxes = self.detector.run(frames)
        print("detector", time.time() - tik)

        tik = time.time()
        masks = self.segmentator.run(global_bbox, frames)
        print("segmentator", time.time() - tik)

        tik = time.time()
        coarse_contours, coarse_keypoints = self.coarse_analyzer.run(masks)
        print("coarse keypoints", time.time() - tik)

        tik = time.time()
        finetuned_contours, finetuned_keypoints = self.fine_analyzer.run(
            frames, coarse_contours, coarse_keypoints
        )
        print("fine_analyzer", time.time() - tik)

        tik = time.time()
        (initial_pivot_sequence,
         tracked_pivot_sequence,
         smoothed_pivot_sequence) = self.pivot_extractor.run(frames, metadata,
                                                             finetuned_contours,
                                                             finetuned_keypoints)
        print("pivot_extractor", time.time() - tik)

        extended_contours = self.__extend_contours(frames[0], finetuned_contours, smoothed_pivot_sequence)
        # for idx in range(len(frames)):
        #     image = cv2.drawContours(frames[idx], [extended_contours[idx]], 0, (0, 0, 255))
        #     image = cv2.drawContours(image, [finetuned_contours[idx]], 0, (0, 0, 255))
        #     for point in smoothed_pivot_sequence[idx]:
        #         image = cv2.circle(image, (point[0], point[1]), 3, (255, 0, 0))
        #     points = np.array(smoothed_pivot_sequence[idx])
            # center_point = np.mean(points, axis=0)
            # image = cv2.circle(image, (int(center_point[0]), int(center_point[1])), 3, (255, 0, 0))

            # cv2.imshow("contour", image)
            # cv2.waitKey(0)

        tik = time.time()
        EF, EF_components, GLS, GLS_components = self.coefficients_estimator.run(metadata, finetuned_contours,
                                                                                 smoothed_pivot_sequence)
        print("cooffice_estimator", time.time() - tik)

        results = {"chamber": chamber, "ef": EF, "gls": GLS,
                   "ef_frames": EF_components, "gls_frames": GLS_components,
                   "inner_contour": [i.squeeze().tolist() for i in finetuned_contours],
                   "outer_contour": [i.squeeze().tolist() for i in extended_contours],
                   "pivot_points": smoothed_pivot_sequence.tolist(),
                   }
        return results


if __name__ == '__main__':
    runtime = ModelRuntime()
    metadata = {'siuid': '1.2.840.113663.1500.244401949268962451674789364464322408',
                'sopiuid': '1.2.840.113663.1500.246186806691199095732815633865241202',
                'frame_time': 0.019518, 'x_scale': 0.03075974343497136,
                'y_scale': 0.03075974343497136, 'heart_rate': "115", 'window': 29}
    for i in range(2):
        runtime.predict("./tmp/sample_dicom.mp4", metadata)
