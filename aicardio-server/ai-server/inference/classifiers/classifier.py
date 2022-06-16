import cv2

import torch
import torch.nn.functional as F

from inference.interfaces import InferenceStep
from inference.config import DEFAULT
from inference.utils.models import load_model_from_module


__all__ = ["ChamberClassifier"]


class ChamberClassifier(InferenceStep):
    r"""DICOM chamber classifier"""
    def __init__(self, config, logger=DEFAULT.logger):
        r"""

        Args:
            config (EasyDict): Configuration of chamber classifier
                .module_name (str): Name of the module containing the classifier
                    class
                .model_class (str): Name of the model class, i.e. a child class
                    of torch.nn.Module
                .model_weights (str): Path to .pth file containing model weights
                    The state dict in the .pth file must contain a key "model"
                    containing the pretrained weights of the classifier
                .image_size (int): Input image size, i.e. input images will be
                    resized to size (image_size, image_size) before passing into
                    the classifier
                .classifier_batch_size (int): Batch size for running classifier
                .device (str): PyTorch device code, e.g. "cuda:0"
        """
        super(ChamberClassifier, self).__init__("ChamberClassifier", logger)
        self.config = config
        self.model = self.__load_model(config.module_name,
                                       config.model_class,
                                       config.model_weights,
                                       config.device)
        self.device = config.device
        self.image_size = config.image_size
        self.idx2chamber = {0: "2C", 1: "3C", 2: "4C", 3: "none"}

    def __load_model(self, module_name, model_class,
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

    def _process(self, item):
        frames = item["frames"]
        chamber = self.run(frames)
        item["chamber"] = chamber 
        return item

    def run(self, frames):
        r"""Run chamber classifier

        This is the public method used for quick testing purpose

        Args:
            frames (np.array): DICOM frames read from pydicom dataset. This is
                a np.array of shape (n_frames, h, w, c)
        Returns:
            str: Chamber of the DICOM video, i.e. one of the following options
                2C, 3C, 4C, or none
        """
        frames = [self.__preprocess(frame) for frame in frames]
        frame_chunks = self.__split_frames_into_chunks(frames)
        with torch.no_grad():
            pred = torch.cat(
                [self.model(chunk.to(self.device)) 
                 for chunk in frame_chunks],
                dim=0
            )
            pred = torch.mean(F.softmax(pred, dim=1), dim=0)
            pred = torch.argmax(pred).item()
        chamber = self.idx2chamber[pred]
        return chamber

    def __preprocess(self, frame):
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = frame.transpose((2, 0, 1)) / 255.0
        frame = frame[None, ...]
        frame = torch.from_numpy(frame).float()
        return frame

    def __split_frames_into_chunks(self, frames):
        frame_chunks = [
            torch.cat(frames[i:i+self.config.batch_size], dim=0)
            for i in range(0, len(frames), self.config.batch_size)
        ]
        return frame_chunks
    
    def _validate_outputs(self, item):
        if "chamber" not in item:
            item["is_valid"] = False
            self.logger.error("No chamber returned")

    def _visualize(self, item):
        return item

    def _log(self, item):
        chamber = item["chamber"]
        self.logger.info(f"\tChamber: {chamber}")
