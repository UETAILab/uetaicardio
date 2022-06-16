import numpy as np
import cv2
import pydicom as dcm

from inference.interfaces import InferenceStep
from inference.config import DEFAULT
import inference.utils as utils


__all__ = ["DICOMReader"]


class DICOMReader(InferenceStep):
    r"""DICOM frame reader"""
    def __init__(self, config, logger=DEFAULT.logger):
        r"""Initialize a DICOM reader

        Args:
            config (EasyDict-like): Configuration for data reader.
                .target_size (tuple): Target size of system input frames. This
                    is a tuple of the form (w, h)
                .default_fps (float): Default FPS.
                .default_heart_rate (float): Default heart rate.
                .window_scale (float): Default window scale.
            logger (logging.Logger): Logger used to display messages.
        """
        super(DICOMReader, self).__init__("DICOMReader", logger)
        self.config = config
        self.frame_extractor = FramesExtractor(self.config.target_size)
        self.metadata_extractor = MetadataExtractor(
            self.config.default_fps, 
            self.config.default_heart_rate, 
            self.config.window_scale
        )
    
    def _validate_inputs(self, item):
        if "dicom_path" not in item:
            item["is_valid"] = False
            self.logger.error("No DICOM path to process")
        else:
            item["is_valid"] = True 
    
    def _process(self, item):
        dicom_path = item["dicom_path"]
        frames, scale, metadata = self.run(dicom_path)
        item.update({"frames": frames, 
                     "inputs_to_original": scale,
                     "metadata": metadata})
        return item

    def run(self, dicom_path):
        r"""Read data from a DICOM file

        This is the public method used for quick testing purpose

        Args:
            dicom_path (str): Path to DICOM
        Returns:
            np.array: Extracted BGR frames from DICOM dataset. This is a 
                np.array of shape (n_frames, h, w, 3).
            float: The scale used to scale original frames' shape to target 
                frames' shape.
            dict: Metadata extracted from DICOM dataset, which consists of
                .siuid (str): SIUID of the DICOM file.
                .sopiuid (str): SOPIUID of the DICOM file.
                .frame_time (float): Duration of a frame in seconds.
                .x_scale (float): Pixel-to-real x-scale
                .y_scale (float): Pixel-to-real y-scale
                .heart_rate (float): Heart rate.
                .window (int): Number of frames per heart cycle.
        """
        dataset = dcm.read_file(dicom_path)
        frames, scale = self.frame_extractor(dataset)
        metadata = self.metadata_extractor(dataset, scale)
        return frames, scale, metadata
    
    def _validate_outputs(self, item):
        if "frames" not in item:
            item["is_valid"] = False
            self.logger.error("Cannot read frames!")
    
    def _visualize(self, item):
        if "visualize" not in item:
            item["visualize"] = {}
        item["visualize"][self.name] = []
        for frame in item["frames"]:
            item["visualize"][self.name].append(frame)
        return item
    
    def _log(self, item):
         if "metedata" in item:
            metadata = item["metadata"]
            self.logger.info(f"\tMetadata: {metadata}")


class FramesExtractor(object):
    r"""Frame extractor used to extract frames from a DICOM dataset"""
    def __init__(self, target_size):
        r"""Initialize a frame extractor

        Args:
            target_size (tuple): Tuple of two integers, which are target (w, h) 
                of the extracted frames.
        """
        self.target_size = target_size 

    def __call__(self, dataset):
        r"""Extract frames from a DICOM dataset.

        Args:
            dataset (pydicom.dataset.FileDataset): DICOM dataset, from which 
                frames are extracted.
        Returns:
            np.array: Extracted BGR frames from DICOM dataset. This is a 
                np.array of shape (n_frames, h_target, w_target, 3).
            float: The scale used to scale original frames to target frames.
        """
        frames = dataset.pixel_array
        frames = self.__expand_dim(frames)
        resized_frames = np.array([
            utils.images.resize(frame, self.target_size) 
            for frame in frames
        ])
        scale = resized_frames.shape[1] / frames.shape[1]
        return resized_frames, scale
    
    def __expand_dim(self, frames):
        if len(frames.shape) == 3:
            if frames.shape[-1] != 3:
                frames = np.repeat(frames[..., None], 3, axis=-1)
            else:
                frames = frames[None, ...]
        if len(frames.shape) == 2:
            frames = np.repeat(frames[None, ..., None], 3, axis=-1)
        return frames


class MetadataExtractor(object):
    r"""Metadata extractor to extract required metadata from a DICOM dataset"""
    def __init__(self, default_fps, default_heart_rate, window_scale):
        self.default_fps = default_fps
        self.default_heart_rate = default_heart_rate
        self.window_scale = window_scale

    def __call__(self, dataset, size_scale):
        r"""Extract frames from a DICOM dataset.

        Args:
            dataset (pydicom.dataset.FileDataset): DICOM dataset, from which 
                frames are extracted.
            size_scale (float): The scale used to scale original frames to target 
                frames. See FrameExtractor for more details.
        Returns:
            dict: Metadata extracted from DICOM dataset, which consists of
                .siuid (str): SIUID of the DICOM file.
                .sopiuid (str): SOPIUID of the DICOM file.
                .frame_time (float): Duration of a frame in seconds.
                .x_scale (float): Pixel-to-real x-scale
                .y_scale (float): Pixel-to-real y-scale
                .heart_rate (float): Heart rate.
                .window (int): Number of frames per heart cycle.
        """
        siuid, sopiuid = self.__extract_uids(dataset)
        frame_time = self.__extract_frame_time(dataset)
        x_scale, y_scale = self.__extract_xy_scales(dataset, size_scale)
        heart_rate = self.__extract_heart_rate(dataset)
        window = int(self.window_scale * ((60/heart_rate) / frame_time))
        metadata = {
            "siuid": siuid, "sopiuid": sopiuid,
            "frame_time": frame_time,
            "x_scale": x_scale, "y_scale": y_scale,
            "heart_rate": heart_rate,
            "window": window
        }
        return metadata

    def __extract_uids(self, dataset):
        try:
            siuid = dataset[0x20, 0xD].value
        except:
            siuid = None
        try:
            sopiuid = dataset[0x8, 0x18].value
        except:
            sopiuid = None
        return siuid, sopiuid

    def __extract_frame_time(self, dataset):
        r"""Extract duration (in seconds) for each frame of the DICOM dataset"""
        try:
            ft = dataset[0x18, 0x1063].value
            return ft / 1000.0
        except:
            pass
        try:
            fps = dataset[0x18, 0x0040].value
            return 1.0 / fps
        except:
            pass
        try:
            fps = dataset[0x7fdf, 0x1074].value
            return 1.0 / fps
        except:
            return 1.0 / self.default_fps

    def __extract_xy_scales(self, dataset, size_scale):
        try:
            x_scale = dataset[0x18, 0x6011].value[0][0x18, 0x602c].value / size_scale 
        except:
            x_scale = 0.02867977775665749 

        try:
            y_scale = dataset[0x18, 0x6011].value[0][0x18, 0x602e].value / size_scale
        except:
            y_scale = 0.02867977775665749
        return x_scale, y_scale

    def __extract_heart_rate(self, dataset):
        r"""Extract heart rate for a DICOm dataset"""
        try:
            heart_rate = dataset[0x18, 0x1088].value
            if heart_rate == 0:
                heart_rate = self.default_heart_rate
        except:
            heart_rate = self.default_heart_rate
        return heart_rate
