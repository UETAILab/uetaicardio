import os
import sys
import glob
import time
import cv2
import torch

import numpy as np
from easydict import EasyDict
import torch.nn.functional as F

from echols.yolov3.utils.utils import *
from echols.yolov3.models import Darknet
from echols.log import logger


class YOLOv3LVDetector:
    def __init__(self, config, ckpt,
                input_size=416,
                batch_size=1,
                device='cpu',
                **kwargs):
        self.model = Darknet(config, input_size).to('cpu')
        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
        self.model.to(device).eval()
        self.input_size = input_size
        self.device = device
        self.batch_size = batch_size
        self.config = EasyDict(locals())

    def detect(self, images):
        """images (ndarray): BGR Images to be detected, i.e. NumPy array of shape (n, h, w, c)
        Returns:
            (list): A list of of ndarrays of shape (n_bboxes, 5), i.e. the detected bounding boxes
                Bounding boxes of each image, each box is of the form [x1, y1, x2, y2, conf]
                Coordinates are normalized to [0, 1]
        """
        image_size = images.shape
        images = self._preprocess(images)
        outputs = self.model(images)
        bboxes = self._postprocess(outputs, image_size)
        return bboxes

    def _preprocess(self, images):
        def pad_to_square(images):
            _, h, w, _ = images.shape
            dim_diff = np.abs(h - w)
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            pad = ((0, 0), (pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (0, 0), (pad1, pad2), (0, 0))
            images = np.pad(images, pad, "constant")
            return images

        images = images[..., ::-1] / 255.0 # BGR to RGB
        images = pad_to_square(images)
        images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
        images = F.interpolate(images, size=self.input_size, mode="nearest")
        images = images.to(self.device)
        return images

    def _postprocess(self, outputs, image_size):
        outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.0) #OK
        bboxes = []
        for detections in outputs:
            if detections is None:
                bboxes.append(np.array([[]]))
                continue
            detections = self.__rescale_bboxes(detections, self.input_size, image_size[1:3])
            detections = self.__stretch_bboxes(detections)
            detections = detections[:, :5].numpy()
            bboxes.append(detections)
        return bboxes

    def __rescale_bboxes(self, boxes, current_dim, original_shape):
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x
        # Rescale bounding boxes to dimension of original image
        boxes[:, 0] = (boxes[:, 0] - pad_x // 2) / unpad_w
        boxes[:, 1] = (boxes[:, 1] - pad_y // 2) / unpad_h
        boxes[:, 2] = (boxes[:, 2] - pad_x // 2) / unpad_w
        boxes[:, 3] = (boxes[:, 3] - pad_y // 2) / unpad_h
        return boxes

    def __stretch_bboxes(self, bboxes):
        w, h = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]
        bboxes[:, 0] = torch.clamp(bboxes[:, 0] - w*0.25, 0, 1)
        bboxes[:, 1] = torch.clamp(bboxes[:, 1] - h*0.75, 0, 1)
        bboxes[:, 2] = torch.clamp(bboxes[:, 2] + w*0.25, 0, 1)
        bboxes[:, 3] = torch.clamp(bboxes[:, 3] + h*0.25, 0, 1)
        return bboxes


if __name__ == "__main__":
    def rectangle(image, pt1, pt2, color, thickness):
        image[pt1[1]-thickness:pt1[1]+thickness, pt1[0]:pt2[0]] = color
        image[pt2[1]-thickness:pt2[1]+thickness, pt1[0]:pt2[0]] = color
        image[pt1[1]:pt2[1], pt1[0]-thickness:pt1[0]+thickness] = color
        image[pt1[1]:pt2[1], pt2[0]-thickness:pt2[0]+thickness] = color

    detector = YOLOv3LVDetector(config='echols/yolov3/config/yolov3-custom.cfg', ckpt='ckpts/0.7435_yolov3_ckpt_75.pth')

    # read 4 images and initialize model
    paths = glob.glob("echols/yolov3/assets/*.png")
    images = np.array([cv2.resize(cv2.imread(path), (256, 256)) for path in paths[:4]])
    logger.debug(images.shape) # (4, 256, 256, 3)

    bboxes = detector.detect(images)
    logger.debug('detector is ok')
    """
    for i in range(len(images)):
        image = images[i]
        h, w, c = image.shape
        boxes = bboxes[i]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * h
        boxes = boxes.astype(int)

        for box in boxes:
            rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(100)
    """
