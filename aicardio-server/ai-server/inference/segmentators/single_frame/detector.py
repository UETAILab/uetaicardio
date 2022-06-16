import itertools
import numpy as np
import cv2
import torch
from torchvision import transforms

from inference.config import DEFAULT
from inference.interfaces import InferenceStep
from inference.segmentators.single_frame.yolov3 import YOLOv3LVDetector


__all__ = ["SingleFrameDetector"]


class SingleFrameDetector(InferenceStep):
    r"""DICOM single-frame LV detector"""
    def __init__(self, config, logger=DEFAULT.logger):
        r"""DICOM LV detector
        
        Args:
            config (EasyDict): Configuration of detector
                .model_def (str): The .cfg file storing the config of the detector
                .image_size (int): The size of the detector input
                .model_weights (str): Path to detector .pth weight file
                .batch_size (int): The batch size used in YOLOv3 LV detector
                .device (str): Device to run the DICOM detector
        """
        super(SingleFrameDetector, self).__init__("SingleFrameDetector", logger)
        self.config = config
        self.detector = self.__load_detector(
            config.model_def,
            config.image_size,
            config.model_weights,
            config.device
        )
        self.image_size = config.image_size
        self.to_tensor = transforms.ToTensor()
    
    def __load_detector(self, model_def, image_size, model_weights, device="cuda:0"):
        return YOLOv3LVDetector(model_def, 
                                image_size, 
                                model_weights, 
                                device)
    
    def _validate_inputs(self, item):
        if "frames" not in item:
            item["is_valid"] = False
            self.logger.error("No frame to process")
    
    def _process(self, item):
        frames = item["frames"]
        global_bbox, bboxes = self.run(frames)
        item.update({"bbox": global_bbox, "bboxes": bboxes})
        return item

    def run(self, frames):
        r"""Run YOLO LV detector

        This is the public method used for quick testing purpose

        Args:
            frames (np.array): DICOM frames. This is a np.array of shape
                (n_frames, h, w, 3)
        Returns:
            np.array: Absolute coordinates of the global bounding box. This is 
                a np.array of shape (5, ), i.e. format (x1, y1, x2, y2, score).
                For example,
                    [269   0 509 394   0]
            list(np.array): Relative coordinates of bounding boxes of all frames.
                Each element is a np.array of shape (n_bboxes, 5), i.e. format
                (x1, y1, x2, y2, score). For example,
                    [0.33706582 0.         0.60501057 0.644969   0.9811854 ]
                In case no bounding box is detected for a certain frame, the
                output would be np.array([[]])
        """
        bboxes = self.__detect_bboxes(frames)
        global_bbox = self.__get_global_bbox(bboxes)
        if global_bbox is not None:
            global_bbox = self.__rescale_bbox(frames.shape[1:], global_bbox)
        return global_bbox, bboxes
        
    def __detect_bboxes(self, frames):
        bboxes = [
            self.detector.detect(frames[i:i+self.config.batch_size]) 
            for i in range(0, len(frames), self.config.batch_size)
        ]
        bboxes = list(itertools.chain.from_iterable(bboxes))
        return bboxes

    def __get_global_bbox(self, bboxes):
        bboxes = [x for x in bboxes if x.size > 0 and len(x) == 1]
        if len(bboxes) == 0:
            return None
        global_bbox = bboxes[0][0].copy()
        global_bbox[0] = min([x[0][...,0] for x in bboxes])
        global_bbox[1] = min([x[0][...,1] for x in bboxes])
        global_bbox[2] = max([x[0][...,2] for x in bboxes])
        global_bbox[3] = max([x[0][...,3] for x in bboxes])
        return global_bbox
    
    def __rescale_bbox(self, image_shape, global_bbox):
        global_bbox[[0, 2]] *= image_shape[1]
        global_bbox[[1, 3]] *= image_shape[0]
        global_bbox = global_bbox.astype(int)
        return global_bbox

    def _validate_outputs(self, item):
        if "bbox" not in item:
            item["is_valid"] = False
            self.logger.error("No global bounding box returned")
        if "bboxes" not in item:
            item["is_valid"] = False
            self.logger.error("No bounding box returned")
    
    def _visualize(self, item):
        if "visualize" not in item:
            item["visualize"] = {}
        item["visualize"][self.name] = []
        for frame, bbox in zip(item["frames"], item["bboxes"]):
            frame = frame.copy()
            if len(bbox[0]) > 0:
                bbox[..., [0, 2]] *= frame.shape[1]
                bbox[..., [1, 3]] *= frame.shape[0]
                cv2.rectangle(
                    frame, tuple(bbox[0, :2]), tuple(bbox[0, 2:4]),
                    (0, 255, 0), 3
                )
            item["visualize"][self.name].append(frame)
        return item
    
    def _log(self, item):
        bbox = item["bbox"]
        self.logger.info(f"\tGlobal bbox: {bbox}")
