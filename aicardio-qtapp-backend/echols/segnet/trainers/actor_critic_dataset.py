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

class ActorCriticDataset(Dataset):
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
        self.image_size = (data_config.image_size, data_config.image_size)

    def gather_dataset(self):
        self.items = [self.__parse_item_from_img_path(img_path, self.config.datadir) for img_path in glob.glob(os.path.join(self.config.datadir, "images", "*.jpg"))]
        self.items = [item for item in self.items if self.__valid_item(item)]

    def __parse_item_from_img_path(self, img_path, datadir):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        video_id = img_id[:img_id.rfind("_")]
        frame_idx = int(img_id[img_id.rfind("_")+1:])
        msk_path = os.path.join(datadir, f"{img_id}.png")
        if not os.path.exists(msk_path):
            msk_path = None
        return EasyDict(dict(img_path=img_path, msk_path=msk_path, video_id=video_id, frame_idx=frame_idx))
    
    def __valid_item(self, item):
        max_video_path = os.path.join(self.config.datadir, "images", f"{item.video_id}_{item.frame_idx+self.config.n_future_neighbors}.jpg")
        return os.path.exists(max_video_path)
    
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
        return self.__read_images(self.items[idx])
    
    def __read_images(self, item):
        to_tensor = transforms.ToTensor()
        
        orig_imgs, orig_msks, msk_availability = [], [], []
        for i in range(item.frame_idx, item.frame_idx+self.config.n_future_neighbors+1):
            # 1. read images
            img_path = os.path.join(self.config.datadir, "images", f"{item.video_id}_{i}.jpg")
            msk_path = os.path.join(self.config.datadir, "masks", f"{item.video_id}_{i}.png")
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            orig_msk = cv2.imread(msk_path) if os.path.exists(msk_path) else np.zeros_like(orig_img)
            orig_imgs.append(orig_img)
            orig_msks.append(orig_msk)
            msk_availability.append(os.path.exists(msk_path))

        # 2. augmentation
        imgs, msks = orig_imgs, orig_msks
        if self.augmentation is not None:
            imgs, msks = self.__do_augmentation(orig_imgs, orig_msks)
        
        # 3. resize
        imgs = [cv2.resize(img, self.image_size, interpolation=cv2.INTER_CUBIC) for img in imgs]
        msks = [cv2.resize(msk, self.image_size, interpolation=cv2.INTER_NEAREST) for msk in msks]
        cv2_msks = [msk.copy() for msk in msks]
        
        vis = [cv2.addWeighted(img, 1, msk, 0.5, 0) for img, msk in zip(imgs, msks)]
        vis = np.concatenate(vis, axis=0)

        # 4. to tensor
        imgs = [to_tensor(img) for img in imgs]
        msks = [to_tensor(msk[...,:1]) for msk in msks]

        # 5. additional process
        if 'additional_process' in self.config and self.config.additional_process is not None:
            results = [self.config.additional_process(img, msk) for img, msk in zip(imgs, msks)]
            imgs = [res[0] for res in results]
            msks = [res[1] for res in results]
        
        imgs = torch.cat([img[None, ...] for img in imgs], axis=0)
        msks = torch.cat([msk[None, ...] for msk in msks], axis=0)
        msk_availability = torch.tensor(msk_availability)
        return EasyDict(dict(
            img_path=item.img_path, imgs=imgs, msks=msks,
            orig_h=orig_h, orig_w=orig_w,
            cv2_msks=cv2_msks, vis=vis, 
            msk_availability=msk_availability
        ))


    def __do_augmentation(self, images, masks):
        '''use albumentations for data augmentation'''
        data = self.__create_augmentation_data(images, masks)
        augmented = self.augmentation(**data)
        return self.__parse_augmentation_result(augmented)
            
    def __create_augmentation_data(self, images, masks):
        data = {}
        for i, (img, msk) in enumerate(zip(images, masks)):
            if i == 0:
                data["image"] = img
                data["mask"] = msk
            else:
                data[f"image{i}"] = img
                data[f"mask{i}"] = msk
        return data
    
    def __parse_augmentation_result(self, augmented):
        images, masks = [], []
        for i in range(self.config.n_future_neighbors+1):
            if i == 0:
                images.append(augmented.get("image", None))
                masks.append(augmented.get("mask", None))
            else:
                images.append(augmented.get(f"image{i}", None))
                masks.append(augmented.get(f"mask{i}", None))
        return images, masks

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
        ], p=p, additional_targets=dict(
            image1="image", image2="image", image3="image", image4="image", image5="image",
            mask1="mask", mask2="mask", mask3="mask", mask4="mask", mask5="mask"
        ))

    def get_datasets(config):
        '''helper function to generate both train and test set from a config'''
        datasets = EasyDict(dict(
            train = ActorCriticDataset.__get_dataset(config.trainset, True, config.image_size, config.n_future_neighbors, config.additional_process),
            test = ActorCriticDataset.__get_dataset(config.testset, False, config.image_size, config.n_future_neighbors, config.additional_process)
        ))
        return datasets
    
    def __get_dataset(datadir, augmentation, image_size, n_future_neighbors, additional_process):
        '''helper function to generate a dataset'''
        config = EasyDict(dict(
            datadir=datadir,
            augmentation=augmentation,
            image_size=image_size,
            n_future_neighbors=n_future_neighbors,
            additional_process=additional_process
        ))
        return ActorCriticDataset(config)

if __name__ == "__main__":
    ds_config = EasyDict(dict(trainset="/data.local/tuannm/data/2020-02-04_2C/train_dev",
                              testset="/data.local/tuannm/data/2020-02-04_2C/test", 
                              image_size=256, 
                              n_future_neighbors=2,
                              additional_process=None))
    ds = ActorCriticDataset.get_datasets(ds_config)
    cv2.imwrite("vis.jpg", ds.train[0].vis)