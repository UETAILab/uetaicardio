import os
import glob
import json
import random
from easydict import EasyDict
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from segmentation.trainers.datasets import SingleFrameDataset


class RightEndoDataset(Dataset):
    r"""Right endothelium dataset for segmentation model training"""
    def __init__(self, data_config):
        ''' Init a SingleFrameDataset with config
        
        config
          .datadir            = data folder with train_dev/ and test/
          .neighbor_size      = number of consecutive frames, on each side, for a single pass
          .augmentation       = set to True for data augmentation
          .image_size         = network input image size
          .additional_process = callable(img, msk) for additional process
        '''
        self.config = data_config
        self.augmentation = self.__strong_aug(p=0.8) if data_config.augmentation else None
        self.neighbor_size = data_config.neighbor_size
        self.image_size = data_config.image_size
        self.to_tensor = transforms.ToTensor()

        self.data_collector = DataCollector()
        self.items = self.data_collector.collect_data(self.config.datadir)
    
    def __getitem__(self, idx):
        images, masks, annotated = self.data_collector.get_video_annotation(
            self.items[idx].json_path,
            self.items[idx].image_dir
        )
        frame_idx = self.__choose_frame(annotated)
        images, mask = self.__get_example(images, masks, frame_idx)
        if self.augmentation is not None:
            images, mask = self.__do_augmentation(images, mask)
        images, mask = self.__preprocess(images, mask)
        return EasyDict(dict(
            images=images, mask=mask, frame_idx=frame_idx
        ))
        
    def __choose_frame(self, annotated):
        available_masks = np.arange(len(annotated))[annotated]
        chosen_frame_idx = random.choice(available_masks)
        return chosen_frame_idx

    def __get_example(self, images, masks, frame_idx):
        collected_images = [
            np.zeros_like(images[frame_idx]) 
            for i in range(-self.neighbor_size, self.neighbor_size+1)
        ]
        for i in range(frame_idx-self.neighbor_size, frame_idx+self.neighbor_size+1):
            if i < 0 or i >= len(images):
                continue
            collected_images[i-frame_idx+self.neighbor_size] = images[i]
        mask = masks[frame_idx] 
        return collected_images, mask

    def __preprocess(self, images, mask):
        images = [cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_CUBIC)
                  for image in images]
        images = torch.cat([self.to_tensor(image)[None, ...] for image in images], 
                          axis=0)
        images = images.permute(1, 0, 2, 3) # (n, c, d, h, w)
        mask = cv2.resize(mask, tuple(self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = self.to_tensor(mask[..., 0])
        if "additional_process" in self.config and self.config.additional_process is not None:
            images, mask = self.config.additional_process(images, mask)
        return images, mask

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
    
    def __len__(self):
        return len(self.items)


class DataCollector(object):
    def __init__(self):
        pass
    
    def collect_data(self, datadir):
        json_paths = glob.glob(os.path.join(datadir, "jsons", "*.json"))
        image_dir = os.path.join(datadir, "images")
        items = []
        for json_path in json_paths:
            items.append(EasyDict(dict(
                json_path=json_path,
                image_dir=image_dir,
            )))
        return items

    def get_video_annotation(self, json_path, image_dir):
        points, boundaries = self.__load_annotation(json_path)
        image_paths = self.__get_image_paths(json_path, image_dir, len(points))
        images = self.__load_images(image_paths)
        annotated_data = [self.__draw_right_endo(i, image, point, boundary)
                    for i, (image, point, boundary) in enumerate(zip(images, points, boundaries))]
        images = [datum[0] for datum in annotated_data]
        masks = [datum[1] for datum in annotated_data]
        annotated = [datum[2] for datum in annotated_data]
        return images, masks, annotated
    
    def __load_images(self, image_paths):
        images = [cv2.imread(path) for path in image_paths]
        return images
    
    def __get_image_paths(self, annotation_path, image_dir, n_images):
        annotation_filename = os.path.basename(annotation_path)
        case_id = os.path.splitext(annotation_filename)[0]
        image_paths = [os.path.join(image_dir, f"{case_id}_{i}.jpg") 
                       for i in range(n_images)]
        return image_paths
    
    def __load_annotation(self, path):
        with open(path, "r") as f:
            annotation = json.load(f)
        points = annotation["point"]["frames"]
        boundaries = annotation["boundary"]["frames"]
        return points, boundaries
    
    def __draw_right_endo(self, idx, image, point, boundary):
        mask = np.zeros_like(image)
        if len(point) != 7 or len(boundary) == 0:
            return image, mask, False
        h, w, _ = image.shape
        pivots = self.__convert_points_to_array(point, w, h)
        contour = self.__convert_points_to_array(boundary[0], w, h)
        peak_idx = self.__proj_point_onto_contour(pivots[3], contour)
        right_idx = self.__proj_point_onto_contour(pivots[-1], contour)
        
        for i in range(min(peak_idx, right_idx), max(peak_idx, right_idx)):
            x1, y1 = int(contour[i, 0]), int(contour[i, 1])
            x2, y2 = int(contour[i+1, 0]), int(contour[i+1, 1])
            cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.circle(mask, (x1, y1), 2, (255, 255, 255), 2)
            cv2.circle(mask, (x2, y2), 2, (255, 255, 255), 2)
        return image, mask, True
    
    def __convert_points_to_array(self, points, w, h):
        points = np.array([[pt["x"], pt["y"]] for pt in points])
        points[:, 0] *= w
        points[:, 1] *= h
        return points

    def __proj_point_onto_contour(self, point, contour):
        dist = np.sum((contour - point)**2, axis=-1)
        return np.argmin(dist)


if __name__ == "__main__":
    config = EasyDict(dict(
        datadir="/data.local/giangh/pipeline/data/2020-03-03_4C/train_dev",
        neighbor_size=8,
        augmentation=False,
        image_size=(256, 256),
    ))
    dataset = RightEndoDataset(config)
    for idx,item in enumerate(dataset):
        print(idx)
        mask = np.uint8(item.mask.numpy()[0]*255.0)
        cv2.imwrite(f'tmp/right_endo_mask/{idx}.jpg',mask)
        print(item.mask.shape)
