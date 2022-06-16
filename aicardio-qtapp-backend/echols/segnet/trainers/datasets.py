import os
import glob
from easydict import EasyDict
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import torch

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
from segmentation.misc.inner_lv_augmentator import InnerLVAugmentator


class SingleFrameDataset(Dataset):
    '''Single-frame dataset for segmentation model training'''
    def __init__(self, data_config):
        ''' Init a SingleFrameDataset with config
        
        config
          .datadir            = data folder with train_dev/ and test/
          .augmentation       = set to True for data augmentation
          .image_size         = network input image size
          .additional_process = callable(img, msk) for additional process
        '''

        self.config = data_config
        self.gather_dataset()
        self.augmentation = self.__strong_aug(p=0.8) if data_config.augmentation else None
        
        inner_lv_augmentator_config = EasyDict(dict(
            datadir="tmp/inner_lv_masks/train",
            prob=0.5,
            max_angle=10,
            dist_scale=1.5,
            overlay=0.7
        ))
        self.inner_lv_augmentator = InnerLVAugmentator(inner_lv_augmentator_config)
        self.image_size = (data_config.image_size, data_config.image_size)

    def gather_dataset(self):
        '''set self.items equal data files [EasyDict(img_path, msk_path)]'''
        self.items = []
        for msk_path in glob.glob(os.path.join(self.config.datadir, "masks", "*.png")):
            # print(msk_path)
            bname = os.path.basename(msk_path)[:-4]
            img_path = os.path.join(self.config.datadir, "images", f"{bname}.jpg")
            if os.path.isfile(img_path):
                self.items.append(EasyDict(dict(img_path=img_path, msk_path=msk_path)))

    def __len__(self):
        '''return number of data items'''
        return len(self.items)

    def __getitem__(self, idx):
        '''return an item by its index
        
        Steps:
        1. read image, mask into cv2 BGR format
        2. do augmentation
        3. resize to self.image_size set by config
        4. use torchvision.transforms.ToTensor
        5. additional process (set by config)
        
        return: EasyDict(img_path, msk_path, orig_w, orig_h, aug_img, aug_msk, aug_blend, img, msk, cv2_msk)
        '''
        
        # 1. read the images
        item = self.items[idx]
        orig_img, orig_msk = cv2.imread(item.img_path), cv2.imread(item.msk_path)
        img, msk = orig_img, orig_msk
        to_tensor = transforms.ToTensor()
        # 2. augmentation
        if self.augmentation is not None:
            img, msk = self.__do_augmentation(img, msk)
        # 3. resize
        img, msk = cv2.resize(img, self.image_size, interpolation=cv2.INTER_CUBIC),\
                   cv2.resize(msk, self.image_size, interpolation=cv2.INTER_NEAREST)
        cv2_msk = msk.copy() # for computing weights in Hausdorff distance loss
        blend = cv2.addWeighted(img, 0.8, msk, 0.2, 0)
        aug_img, aug_msk, aug_blend = to_tensor(img), to_tensor(msk), to_tensor(blend)
        # 4. to tensor
        img, msk = to_tensor(img), to_tensor(msk[...,0])
        # 5. additional process
        if 'additional_process' in self.config and self.config.additional_process is not None:
            img, msk = self.config.additional_process(img, msk)
        
        return EasyDict(dict(
            img_path=item.img_path, msk_path=item.msk_path,
            # orig_img=orig_img, orig_msk=orig_msk,
            orig_w=orig_img.shape[1], orig_h=orig_img.shape[0],
            aug_img=aug_img, aug_msk=aug_msk, aug_blend=aug_blend,
            img=img, msk=msk, cv2_msk=cv2_msk
        ))

    def __do_augmentation(self, image, mask):
        '''use albumentations for data augmentation'''
        image, mask = self.inner_lv_augmentator.augment(image, mask)
        data = {"image": image, "mask": mask}
        augmented = self.augmentation(**data)
        return augmented['image'], augmented['mask']

    def __strong_aug(self, p=0.5):
        '''preset augmentation'''
        return Compose([
            # RandomRotate90(),
            # Flip(),
            # Transpose(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
        ], p=p)
        
    def summary(self):
        '''print a summary of loaded dataset for debugging'''
        print("dataset", self.config)
        print("n_item", len(self))
        loader = DataLoader(self, batch_size=4, shuffle=True)
        for batch in loader:
            print(batch['img_path'], batch['msk_path'], batch['img'].shape, batch['msk'].shape)
            tensor = torch.cat([batch['aug_img'], batch['aug_msk'], batch['aug_blend']], dim=0)
            print("tensor", tensor.shape)
            save_image(tensor, "tmp/batch.png", nrow=4)
            break

    def get_datasets(config):
        '''helper function to generate both train and test set from a config'''
        datasets = EasyDict(dict(
            train = SingleFrameDataset.__get_dataset(config.trainset, True, config.image_size),
            test = SingleFrameDataset.__get_dataset(config.testset, False, config.image_size)
        ))
        return datasets
    
    def __get_dataset(datadir, augmentation, image_size, additional_process=None):
        '''helper function to generate a dataset'''
        config = EasyDict(dict(
            datadir=datadir,
            augmentation=augmentation,
            image_size=image_size,
            additional_process=additional_process
        ))
        return SingleFrameDataset(config)

class HDSingleFrameDataset(SingleFrameDataset):
    '''Single-frame dataset with Hausdorff distance weights for segmentation model training'''
    def __init__(self, data_config):
        ''' Init a HDSingleFrameDataset with config
        
        config
          .datadir            = data folder with train_dev/ and test/
          .augmentation       = set to True for data augmentation
          .image_size         = network input image size
          .additional_process = callable(img, msk) for additional process
        '''

        super().__init__(data_config)
    
    def __getitem__(self, idx):
        '''return an item by its index, adding Hausdorff distance weights'''
        item = super().__getitem__(idx)
        item.hd = self.__get_hausdorff_distance(item.cv2_msk)
        item.vertical_weights = self.__get_vertical_weights(item.cv2_msk)
        return item

    def __get_hausdorff_distance(self, cv2_msk):
        '''compute Hausdorff distance weights'''
        # 1. compute mask
        bw = cv2.cvtColor(cv2_msk, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 2. compute distance of mask's inner points
        dist_inner = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
        cv2.normalize(dist_inner, dist_inner, 0, 1.0, cv2.NORM_MINMAX)
        # 3. compute distance of mask's outer points
        dist_outer = cv2.distanceTransform(255-bw, cv2.DIST_L2, 3)
        cv2.normalize(dist_outer, dist_outer, 0, 1.0, cv2.NORM_MINMAX)
        # 4. fuse inner and outer distance and normalize
        dist = dist_inner + dist_outer
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        return np.float32(dist)
    
    def __get_vertical_weights(self, cv2_msk, max_weights=5):
        '''compute weight (increase along depth)'''
        bw = cv2.cvtColor(cv2_msk, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ys, xs = np.where(bw > 0)
        
        ymin, ymax = np.min(ys), np.max(ys)
        inner_weights = np.ones_like(bw, dtype=np.float32)
        base_weights = (2 * (np.arange(ymin, ymax) - (ymin + ymax)/2) / (ymax - ymin))**2
        inner_weights[ymin:ymax] = inner_weights[ymin:ymax] + (max_weights * base_weights)[..., None]
        inner_weights = (bw > 0).astype(float) * inner_weights
        inner_weights += 1
        return inner_weights
    
    def get_datasets(config):
        '''helper function to generate both train and test set from a config'''
        datasets = EasyDict(dict(
            train = HDSingleFrameDataset.__get_dataset(config.trainset, True, config.image_size, config.additional_process),
            test = HDSingleFrameDataset.__get_dataset(config.testset, False, config.image_size, config.additional_process)
        ))
        return datasets
    
    def __get_dataset(datadir, augmentation, image_size, additional_process):
        '''helper function to generate a dataset'''
        config = EasyDict(dict(
            datadir=datadir,
            augmentation=augmentation,
            image_size=image_size,
            additional_process=additional_process
        ))
        return HDSingleFrameDataset(config)

def single_image_normalize(img, msk):
    img = (img - img.mean(dim=(1, 2))[:, None, None]) / img.std(dim=(1, 2))[:, None, None]
    return img, msk 
    
def torch_hub_normalize(img, msk):
    hub_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img[[2,1,0],:,:] # convert to RGB and normalize
    img = hub_normalize(img)
    return img, msk
    
if __name__ == '__main__':
    ds = HDSingleFrameDataset.get_dataset("/data.local/tuannm/data/2020-02-04_2C/train_dev", True, 256)
    print(len(ds))
    print(dir(ds[0]))
    print(ds[0].hd)
