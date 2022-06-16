import os
import glob
from easydict import EasyDict

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class StandardDataset(Dataset):
    def __init__(self, config):
        r"""Standard dataset for chamber classification. Re-implement
        ImageFolder dataset of torchvision

        Args:
            config (EasyDict): Configuration for classification dataset
                .root (str): Data root
                .augment (callable): Augmentation function. Default None
                .preprocess (callable): Preprocess function. Default None
        """
        self.config = config
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
        items = [EasyDict(dict(image_path=path, class_id=class_id)) for path in glob.glob(
                os.path.join(self.config.root, _class, "*.jpg")
        )]
        return items

    def __getitem__(self, idx):
        image = cv2.imread(self.items[idx].image_path)
        label = self.items[idx].class_id
        
        if self.augment:
            image = self.augment(image)
        if self.preprocess:
            image = self.preprocess(image)
        
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        label = torch.tensor(label).long()
        return EasyDict(dict(
            image_path = self.items[idx].image_path,
            image=image, label=label
        ))

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    config = EasyDict(dict(
        root="/data.local/giangh/pipeline/data/classification/train",
        augment=None, preprocess=None
    ))
    dataset = StandardDataset(config)

    idx = np.random.randint(low=0, high=len(dataset))
    item = dataset[idx]
    image, label = item["image"], item["label"]

    image = image.numpy()
    cv2.imwrite(f"{label}.jpg", image)
