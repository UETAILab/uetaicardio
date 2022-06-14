import os
import json
import cv2
import pydicom
import argparse
import numpy as np

from easydict import EasyDict
from pydicom.pixel_data_handlers.util import convert_color_space
from torch.utils.data import Dataset
from echols.log import logger


class FramesExtractor:
    def __init__(self, target_size=(800, 600)):
        self.w_target, self.h_target = target_size

    def extract_frames(self, dataset: pydicom.dataset.FileDataset):
        r"""Read Dicom and return list of BGR images (n, h, w, 3) and a Float value used as scale"""
        frames = dataset.pixel_array
        frames = convert_color_space(frames, 'YBR_FULL', 'RGB')
        if len(frames.shape) == 3 and frames.shape[-1] != 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        resized_frames = self.__pad_and_resize(frames)
        scale = resized_frames.shape[1] / frames.shape[1]
        return resized_frames, scale

    def __pad_and_resize(self, frames):
        r"""Pad and resize frames to target size and keep aspect ratio"""

        def get_pad(frames):
            _, h, w, _ = frames.shape
            w_pad = int(max(w, h * self.w_target / self.h_target))
            h_pad = int(max(h, w * self.h_target / self.w_target))
            pad_left = (w_pad - w) // 2
            pad_right = w_pad - w - pad_left
            pad_top = (h_pad - h) // 2
            pad_bottom = h_pad - h - pad_top
            return ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

        pad = get_pad(frames)
        frames = [np.pad(frame, pad, mode="constant") for frame in frames]
        frames = np.array([cv2.resize(frame, (self.w_target, self.h_target)) for frame in frames])
        return frames

class MetadataExtractor:
    def __init__(self, default_fps=30, default_heart_rate=75, window_scale=1.1):
        self.default_fps = default_fps
        self.default_heart_rate = default_heart_rate
        self.window_scale = window_scale

    def extract_metadata(self, dataset, size_scale):
        r"""
        Args:
            dataset (pydicom.dataset.FileDataset): DICOM dataset, from which frames are extracted.
            size_scale (float): The scale used to scale original frames to target frames. See FrameExtractor for more details.
        Returns:
            Metadata extracted from DICOM dataset, which consists of
                .frame_time (float): Duration of a frame in seconds.
                .x_scale (float): Pixel-to-real x-scale
                .y_scale (float): Pixel-to-real y-scale
                .heart_rate (float): Heart rate.
                .window (int): Number of frames per heart cycle.
        """
        frame_time = self.__extract_frame_time(dataset)
        x_scale = dataset[0x18, 0x6011].value[0][0x18, 0x602c].value / size_scale
        y_scale = dataset[0x18, 0x6011].value[0][0x18, 0x602e].value / size_scale
        heart_rate = self.__extract_heart_rate(dataset)
        window = int(self.window_scale * ((60/heart_rate) / frame_time))
        metadata = EasyDict(dict(
            frame_time=frame_time,
            x_scale=x_scale,
            y_scale=y_scale,
            heart_rate=heart_rate,
            window=window
        ))
        return metadata

    def __extract_frame_time(self, dataset):
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

    def __extract_heart_rate(self, dataset):
        try:
            heart_rate = dataset[0x18, 0x1088].value
        except:
            heart_rate = self.default_heart_rate
        return heart_rate

class DICOMDataset(Dataset):
    def __init__(self, dicom_path,
                       frame_start=0,
                       target_size=(800, 600),
                       fps=30, heart_rate=75,
                       window_scale=1.1,
                       case_idx=0,
                       **kwargs):
        self.config = EasyDict(locals())
        logger.debug("Dataset loader'sconfig:")
        logger.debug(self.config)
        self.frame_extractor = FramesExtractor(target_size)
        self.metadata_extractor = MetadataExtractor(
            fps,
            heart_rate,
            window_scale=window_scale,
        )
        self.__gather_dataset()

    def __gather_dataset(self):
        dataset = pydicom.read_file(self.config.dicom_path)
        self.items, self.scale = self.frame_extractor.extract_frames(dataset)
        self.metadata = self.metadata_extractor.extract_metadata(dataset, self.scale)
        #self.items = self.items[self.config.frame_start:self.config.frame_start+self.metadata.window]
        logger.info(f"Number of frame: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image = self.items[idx]
        return EasyDict(dict(image=image, w=image.shape[1], h=image.shape[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_path", type=str, help="DICOM path")
    parser.add_argument("--viz_dir", type=str, default=None, \
            help="visualization directory (not None then visualize outputs)")
    args = parser.parse_args()
    dataset = DICOMDataset(args.dicom_path)

    os.makedirs(os.path.join(args.viz_dir), exist_ok=True) if args.viz_dir else None
    print(f"DICOM file: {args.dicom_path}")
    print(f"Number of frames: {len(dataset)}")
    for i, item in enumerate(dataset):
        print(f"\tFrame shape {item.image.shape}")
        cv2.imwrite(os.path.join(args.viz_dir, f"{i}.jpg"), item.image) if args.viz_dir else None
        cv2.imshow('a', item.image)
        cv2.waitKey(30)
