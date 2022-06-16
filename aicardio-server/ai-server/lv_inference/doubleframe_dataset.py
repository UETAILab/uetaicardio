import os
import json
import numpy as np
import cv2
import pydicom
from easydict import EasyDict
from torch.utils.data import Dataset
import argparse


class DoubleFrameDICOMDataset(Dataset):
    r"""DICOM dataset used for EF and GLS inference"""
    
    def __init__(self, data_config):
        r"""
        Args:
            data_config (EasyDict): The configuration of the DICOM dataset. This is a dict with keys:
                .dicom_path (str): Path to the DICOM file.
                .target_size (tuple): Target size of input images to the system. This is a tuple of two integers, i.e. (w, h).
                .default_heart_rate (float): Default heart rate in case the DICOM file does not include heart rate.
                .default_fps (int): Default FPS in case the DICOM file does not include FPS.
                .frame_start (int): Index of starting frame for EF and GLS computation.
        """        
        self.config = data_config
        self.config.frame_start *= 2
        self.config.default_fps *= 2
        self.config.default_heart_rate /= 2

        self.frame_extractor = DoubleFramesExtractor(self.config.target_size)
        self.metadata_extractor = MetadataExtractor(
            self.config.default_fps,
            self.config.default_heart_rate,
            window_scale=1.1,
        )
        self.__gather_dataset()

    def __gather_dataset(self):
        dataset = pydicom.read_file(self.config.dicom_path)
        self.items, self.scale = self.frame_extractor.extract_frames(dataset)
        self.metadata = self.metadata_extractor.extract_metadata(dataset, self.scale)
        self.items = self.items[self.config.frame_start:self.config.frame_start+self.metadata.window]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image = self.items[idx]
        return EasyDict(dict(image=image, w=image.shape[1], h=image.shape[0]))


class DoubleFramesExtractor:
    r"""Frame extractor used to extract frames from a DICOM dataset"""
    def __init__(self, target_size):
        r"""

        Args:
            target_size (tuple): Tuple of two integers, which are target (w, h) of the extracted frames.
        """
        self.w_target, self.h_target = target_size

    def extract_frames(self, dataset):
        r"""Extract frames from a DICOM dataset.

        Args:
            dataset (pydicom.dataset.FileDataset): DICOM dataset, from which frames are extracted.
        Returns:
            np.array: Extracted BGR frames from DICOM dataset. This is a np.array of shape (n_frames, h_target, w_target, 3).
            float: The scale used to scale original frames to target frames.
        """
        frames = dataset.pixel_array
        if len(frames.shape) == 3 and frames.shape[-1] != 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        frames = self.__double_frames(frames)
        resized_frames = self.__pad_and_resize(frames)
        scale = resized_frames.shape[1] / frames.shape[1]
        return resized_frames, scale

    def __double_frames(self, frames, n=2):
        doubled_frames = [frames[0]]
        for i in range(1, len(frames)):
            items = []
            for j in range(1, n+1):
                w = j/n
                item = cv2.addWeighted(frames[i-1], 1-w, frames[i], w, 0)
                items.append(item)
            doubled_frames.extend(items)
        doubled_frames = np.array(doubled_frames)
        return doubled_frames

    def __pad_and_resize(self, frames):
        r"""Pad and resize frames to target size to keep aspect ratio"""
        pad = self.__get_pad(frames)
        frames = [np.pad(frame, pad, mode="constant") for frame in frames]
        frames = np.array([cv2.resize(frame, (self.w_target, self.h_target)) for frame in frames])
        return frames
    
    def __get_pad(self, frames):
        r"""Get padding for __pad_and_resize()"""
        _, h, w, _ = frames.shape
        w_pad = int(max(w, h * self.w_target / self.h_target))
        h_pad = int(max(h, w * self.h_target / self.w_target))
        pad_left = (w_pad - w) // 2
        pad_right = w_pad - w - pad_left
        pad_top = (h_pad - h) // 2
        pad_bottom = h_pad - h - pad_top
        return ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))


class MetadataExtractor:
    r"""Metadata extractor to extract required metadata from a DICOM dataset"""
    def __init__(self, default_fps, default_heart_rate, window_scale):
        self.default_fps = default_fps
        self.default_heart_rate = default_heart_rate
        self.window_scale = window_scale

    def extract_metadata(self, dataset, size_scale):
        r"""Extract frames from a DICOM dataset.

        Args:
            dataset (pydicom.dataset.FileDataset): DICOM dataset, from which frames are extracted.
            size_scale (float): The scale used to scale original frames to target frames. See FrameExtractor for more details.
        Returns:
            EasyDict: Metadata extracted from DICOM dataset, which consists of
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
        r"""Extract duration (in seconds) for each frame of the DICOM dataset"""
        try:
            ft = dataset[0x18, 0x1063].value / 2
            return ft / 1000.0
        except:
            pass
        try:
            fps = 2 * dataset[0x18, 0x0040].value
            return 1.0 / fps
        except:
            pass
        try:
            fps = 2 * dataset[0x7fdf, 0x1074].value
            return 1.0 / fps
        except:
            return 1.0 / self.default_fps

    def __extract_heart_rate(self, dataset):
        r"""Extract heart rate for a DICOm dataset"""
        try:
            heart_rate = dataset[0x18, 0x1088].value / 2
        except:
            heart_rate = self.default_heart_rate
        return heart_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_path", type=str, help="DICOM path")
    parser.add_argument("--visualize_dir", type=str, default=None, help="visualization directory (not None then visualize outputs)")
    args = parser.parse_args()

    case_idx = args.dicom_path[args.dicom_path.find("__")-4:args.dicom_path.find("__")]
    case_name = args.dicom_path[args.dicom_path.find("__")+2:args.dicom_path.rfind("/")]
    with open("bash_scripts/frame_starT_EF_GLS.json") as f:
        frame_start_EF_GLS = json.load(f)
    if case_idx in frame_start_EF_GLS:
        frame_start_EF_GLS = frame_start_EF_GLS[case_idx]["2C"]
    else:
        frame_start_EF_GLS = 0

    data_config  = EasyDict(dict(case_idx=case_idx,
                            dicom_path=args.dicom_path,
                            target_size=(800, 600),
                            default_heart_rate=75.0,
                            default_fps=30,
                            frame_start=frame_start_EF_GLS))
    dataset = DICOMDataset(data_config)
    
    os.makedirs(os.path.join(args.visualize_dir), exist_ok=True)
    print(f"DICOM file: {args.dicom_path}")
    print(f"Number of frames: {len(dataset)}")
    for i, item in enumerate(dataset):
        if i == 5:
            break
        print(f"\tFrame shape {item.image.shape}")
        cv2.imwrite(os.path.join(args.visualize_dir, f"{i}.jpg"), item.image)