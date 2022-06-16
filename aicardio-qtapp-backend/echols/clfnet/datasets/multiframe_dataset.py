#!/data.local/giangh/envs/pipeline/bin/python
import os
import glob
from easydict import EasyDict
import pydicom as dcm
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MultiframeDataset(Dataset):
    def __init__(self, config):
        r"""Multi-frame Dataset for chamber classification. Re-implement
        ImageFolder dataset of torchvision

        Args:
            config (EasyDict): Configuration for classification dataset
                .root (str): Data root, which is a folder of the form
                    |-- root
                        |-- <class 1>
                            |-- <dicom path>
                            |-- <dicom path>
                            |-- ...
                        |-- <class 2>
                            |-- <dicom path>
                            |-- <dicom path>
                            |-- ...
                        |-- ...
                .window_size (int): Number of frames to be taken. This is the
                    maximum number of frames, which a heart cycle can span
                .augment (callable): Augmentation function. Default None
                    Input: list of np arrays
                    Output: list of np arrays
                .preprocess (callable): Preprocess function. Default None
                    Input: list of np arrays
                    Output: list of np arrays
                .is_training (bool): Whether in training mode or not
        """
        self.config = config
        self.window_size = config.window_size
        self.augment = config.augment
        self.preprocess = config.preprocess
        self.items = self.__gather_dataset()

    def __gather_dataset(self):
        self.classes = ["2C", "3C", "4C", "none"]
        self.items = []
        for class_id, _class in enumerate(self.classes):
            class_items = self.__gather_class_items(_class, class_id)
            self.items.extend(class_items)
        return self.items

    def __gather_class_items(self, _class, class_id):
        items = [
            EasyDict(dict(dicom_path=path, class_id=class_id)) 
            for path in glob.glob(os.path.join(self.config.root, _class, "*"))
        ]
        return items

    def __getitem__(self, idx):
        frames = self.__read_frames(self.items[idx].dicom_path)
        label = self.items[idx].class_id
        
        if self.augment:
            frames = self.augment(frames)
        if self.preprocess:
            frames = self.preprocess(frames)
        
        frames = torch.cat([
            torch.from_numpy(frame.transpose((2, 0, 1))[:, None, :, :]).float()
            for frame in frames
        ], dim=1)
        label = torch.tensor(label).long()
        return EasyDict(dict(
            dicom_path=self.items[idx].dicom_path,
            frames=frames, label=label
        ))

    def __read_frames(self, dicom_path):
        dataset = dcm.read_file(dicom_path)
        frames = dataset.pixel_array
        frames = self.__normalize_frames_dim(frames)
        frames = self.__sample_frames(frames)
        return list(frames)
    
    def __normalize_frames_dim(self, frames):
        if len(frames.shape) == 2:
            frames = np.repeat(frames[None, ..., None], 3, axis=-1)
        elif len(frames.shape) == 3:
            if frames.shape[-1] == 3:
                frames = frames[None, ...]
            else:
                frames = np.repeat(frames[..., None], 3, axis=-1)
        return frames

    def __sample_frames(self, frames):
        if len(frames) >= self.window_size:
            if self.config.is_training:
                start_frame = np.random.randint(0, len(frames)-self.window_size+1)
            else:
                start_frame = 0
            frames = frames[start_frame:start_frame+self.window_size]
            return frames
        else:
            padded_frames = np.zeros(
                (self.window_size, frames.shape[1], frames.shape[2], frames.shape[3]), 
                dtype=frames.dtype
            )
            padded_frames[:len(frames)] = frames
            return padded_frames

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    config = EasyDict(dict(
        root="/data.local/giangh/pipeline/data/classification/multi_frame/train",
        window_size=40, augment=None, preprocess=None
    ))
    dataset = MultiframeDataset(config)
    
    idx = np.random.randint(0, len(dataset))
    item = dataset[idx]
    print(item["dicom_path"])
    print(item["frames"].shape)
    print(item["label"])
