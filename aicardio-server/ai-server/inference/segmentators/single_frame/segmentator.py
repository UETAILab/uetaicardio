import numpy as np
import cv2
import torch
from torchvision import transforms

from inference.interfaces import InferenceStep
from inference.config import DEFAULT
from inference.utils.models import load_model_from_module


__all__ = ["SingleFrameSegmentator"]


class SingleFrameSegmentator(InferenceStep):
    r"""DICOM LV segmentator"""
    def __init__(self, config, logger=DEFAULT.logger):
        r"""
        
        Args:
            config (EasyDict): Configuration of segmentator
                .full_image_config.
                    .module_name (str): Path to the module containing the 
                        full-image segmentator class
                    .model_class (str): Name of the segmentator class
                    .model_weights (str): The .pth file containing the weights of 
                        the full-image segmentator
                .cropped_image_config.
                    .module_name (str): Path to the module containing the cropped 
                        segmentator class
                    .model_class (str): Name of the segmentator class
                    .model_weights (str): The .pth file containing the weights of 
                        the cropped segmentator
                .device (str): Device to run the DICOM segmentator
                .image_size (int): Input size of the segmentators
        """
        super(SingleFrameSegmentator, self).__init__("SingleFrameSegmentator", 
                                                     logger)
        self.config = config
        self.full_image_segmentator = self.__load_segmentator(
            config.full_image_config.module_name, 
            config.full_image_config.model_class, 
            config.full_image_config.model_weights,
            config.device
        )
        self.cropped_image_segmentator = self.__load_segmentator(
            config.cropped_image_config.module_name, 
            config.cropped_image_config.model_class, 
            config.cropped_image_config.model_weights,
            config.device
        )
        self.image_size = config.image_size
        self.to_tensor = transforms.ToTensor()
    
    def __load_segmentator(self, module_name, model_class, 
                           model_weights, device="cuda:0"):
        model = load_model_from_module(module_name, model_class, device)
        self.__load_weights(model, model_weights, device)
        model.eval()
        return model
    
    def __load_weights(self, model, model_weights, device):
        checkpoint_data = torch.load(model_weights, map_location=device)
        model.load_state_dict(checkpoint_data["model"])

    def _validate_inputs(self, item):
        if "frames" not in item:
            item["is_valid"] = False
            self.logger.error("No frame to process")
        if "bbox" not in item:
            item["is_valid"] = False
            self.logger.error("No bounding box to process")

    def _process(self, item):
        bbox = item["bbox"]
        frames = item["frames"]
        masks = self.run(bbox, frames)
        item["masks"] = masks
        return item

    def run(self, bbox, frames):
        r"""Segment LV mask from DICOM frames

        This is the public method used for quick testing purpose

        Args:
            bbox (np.array): Absolute coordinates of the global bounding box. 
                This is a np.array of shape (5, ), i.e. format (x1, y1, x2, y2, 
                score). For example,
                    [269   0 509 394   0]
            frames (np.array): DICOM frames. This is a np.array of shape
                (n_frames, h, w, 3)
        Returns:
            list(np.array): List of masks. This is a list of np.arrays, each of 
                which has shape (h, w, 3)
        """
        if bbox is not None:
            segmentator = self.cropped_image_segmentator
        else:
            segmentator = self.full_image_segmentator
        preprocessed_frames = [self.__preprocess(frame, bbox) 
                               for frame in frames]
        masks = [self.__segment_one_frame(segmentator, frame) 
                 for frame in preprocessed_frames]
        masks = [self.__postprocess(mask, frame, bbox) 
                 for frame, mask in zip(frames, masks)]
        return masks
    
    def __preprocess(self, frame, global_bbox):
        if global_bbox is not None:
            frame  = frame[global_bbox[1]:global_bbox[3], 
                           global_bbox[0]:global_bbox[2]]
        frames = cv2.resize(frame, (self.image_size, self.image_size), 
                            interpolation=cv2.INTER_CUBIC)
        frame = self.to_tensor(frame).to(self.config.device)
        return frame
    
    def __segment_one_frame(self, model, frame):
        with torch.no_grad():
            mask = model(frame[None, ...])[0]
            mask = torch.sigmoid(mask)
        return mask
    
    def __postprocess(self, mask, frame, global_bbox):
        mask = mask.cpu().numpy().transpose((1, 2, 0))
        mask = np.uint8(mask * 255)
        mask = np.repeat(mask, 3, axis=-1)
        if global_bbox is not None:
            original_size = (global_bbox[2] - global_bbox[0],
                             global_bbox[3] - global_bbox[1]) 
            cropped_mask = cv2.resize(mask, original_size, 
                                      interpolation=cv2.INTER_NEAREST)
            mask = np.zeros((frame.shape), dtype=np.uint8)
            mask[global_bbox[1]:global_bbox[3], 
                      global_bbox[0]:global_bbox[2]] = cropped_mask
        else:
            original_size = (frame.shape[1], frame.shape[0])
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        return mask
    
    def _validate_outputs(self, item):
        if "masks" not in item:
            item["is_valid"] = False
            self.logger.error("No mask returned")
    
    def _visualize(self, item):
        if "visualize" not in item:
            item["visualize"] = {}
        item["visualize"][self.name] = []
        for frame, mask in zip(item["frames"], item["masks"]):
            frame = frame.copy()
            frame = cv2.addWeighted(frame, 1, mask, 0.75, 0)
            item["visualize"][self.name].append(frame)
        return item
    
    def _log(self, item):
        n_masks = len(item["masks"])
        self.logger.info(f"\tNumber of masks: {n_masks}")
