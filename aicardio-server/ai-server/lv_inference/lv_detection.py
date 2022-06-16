import os, glob, sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from PyTorch_YOLOv3.utils.utils import *
from PyTorch_YOLOv3.models import Darknet

class YOLOv3LVDetector:
    def __init__(self, model_def, image_size=416, ckpt=None, device=torch.device("cpu")):
        self.model = Darknet(model_def, image_size).to(device)
        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt))
        self.model.eval()
        self.image_size = image_size
        self.device = device
    
    def detect(self, images):
        """Detect bounding boxes from a batch of images
        
        Args:
            images (ndarray): BGR Images to be detected, i.e. NumPy array of shape (n, h, w, c)
        Returns:
            (list): A list of of ndarrays of shape (n_bboxes, 5), i.e. the detected bounding boxes
                Bounding boxes of each image, each box is of the form [x1, y1, x2, y2, conf]
                Coordinates are normalized to [0, 1]
        """
        image_size = images.shape
        images = self.__preprocess(images)
        outputs = self.model(images)
        bboxes = self.__postprocess(outputs, image_size)
        return bboxes
    
    def __preprocess(self, images):
        images = self.__pad_to_square(images[..., ::-1] / 255.0)
        images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
        images = F.interpolate(images, size=self.image_size, mode="nearest")
        images = images.to(self.device)
        return images
    
    def __pad_to_square(self, images):
        _, h, w, _ = images.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((0, 0), (pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (0, 0), (pad1, pad2), (0, 0))
        images = np.pad(images, pad, "constant")
        return images
    
    def __postprocess(self, outputs, image_size):
        outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.0) #OK
        bboxes = []
        for detections in outputs:
            if detections is None:
                bboxes.append(np.array([[]]))
                continue
            detections = self.__rescale_bboxes(detections, self.image_size, image_size[1:3])
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

# Demo code
if __name__ == "__main__":
    # misc
    def rectangle(image, pt1, pt2, color, thickness):
        image[pt1[1]-thickness:pt1[1]+thickness, pt1[0]:pt2[0]] = color
        image[pt2[1]-thickness:pt2[1]+thickness, pt1[0]:pt2[0]] = color
        image[pt1[1]:pt2[1], pt1[0]-thickness:pt1[0]+thickness] = color
        image[pt1[1]:pt2[1], pt2[0]-thickness:pt2[0]+thickness] = color
    
    # initialize detector
    detector = YOLOv3LVDetector(config.YOLOv3_MODEL_DEF, 
                                config.YOLOv3_IMAGE_SIZE, 
                                config.YOLOv3_CHECKPOINT, 
                                config.YOLOv3_DEVICE)
    
    # read 4 images and initialize model
    paths = glob.glob("../pipeline_aicardio/PyTorch-YOLOv3/data/valid/images/GEMS_IMG__2010*.jpg")
    images = np.concatenate([cv2.imread(path)[None, ...] for path in paths[:4]], axis=0)
    bboxes = detector.detect(images)
    
    # run detector
    start = time.time()
    for i in range(100):
        bboxes = detector.detect(images)
    print(f"Running time: {time.time() - start} for {100} images")
    
    # visualize result
    for i in range(len(images)):
        image = images[i]
        h, w, c = image.shape
        boxes = bboxes[i]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * h
        boxes = boxes.astype(int)

        for box in boxes:
            rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite(f"test-{i}.jpg", image)