import os
import glob
import random
from easydict import EasyDict
import numpy as np
import cv2

class InnerLVAugmentator:
    def __init__(self, data_config):
        ''' Init a SingleFrameDataset with config
        
        config
          .datadir            = inner lv data folder with images/ and masks/
          .prob (float): probability to augment
          .max_angle (float): max angle to randomly rotate inner lv patch
          .dist_scale (float): how to scale the distance map (then clip it to [0..1])
          .overlay (float): a constant in [0..1], how much to overlay inner lv area onto the original image
        '''
        
        self.config = data_config
        self.gather_dataset()
        self.prob = self.config.prob
        self.max_angle = self.config.max_angle
        self.dist_scale = self.config.dist_scale
        self.overlay = self.config.overlay

    def gather_dataset(self):
        '''set self.items equal data files [EasyDict(img_path, msk_path)]'''
        self.items = []
        for msk_path in glob.glob(os.path.join(self.config.datadir, "masks", "*.png")):
            # print(msk_path)
            bname = os.path.basename(msk_path)[:-4]
            img_path = os.path.join(self.config.datadir, "images", f"{bname}.jpg")
            if os.path.isfile(img_path):
                self.items.append(EasyDict(dict(img_path=img_path, msk_path=msk_path)))
    
    def augment(self, image, mask):
        random_number = np.random.uniform()
        if random_number > self.prob:
            return image, mask

        inner_lv_item = random.choice(self.items)
        inner_lv_image = cv2.imread(inner_lv_item.img_path)
        inner_lv_mask = cv2.imread(inner_lv_item.msk_path)
        binary_inner_lv_mask = self.__convert_mask_to_binary(inner_lv_mask)
        binary_mask = self.__convert_mask_to_binary(mask)

        normalized_inner_lv_image, normalized_inner_lv_mask = self.__normalize_inner_lv_patch(inner_lv_image, inner_lv_mask, image, binary_mask, binary_inner_lv_mask)
        inner_lv_dist_map = self.__get_inner_lv_dist_map(normalized_inner_lv_mask)
        normalized_inner_lv_image = np.uint8(inner_lv_dist_map[..., None] * normalized_inner_lv_image)
        augmented_image = cv2.addWeighted(image, 1, normalized_inner_lv_image, self.overlay, 0)
        
#         #---
#         print(inner_lv_item.img_path)
#         inner_lv_image = cv2.resize(inner_lv_image, (image.shape[1], image.shape[0]))
#         inner_lv_dist_map = np.repeat(inner_lv_dist_map[..., None] * 255, 3, axis=-1).astype(np.uint8)
#         vis = np.concatenate([image, 
#                               augmented_image,
#                               inner_lv_image,
#                               normalized_inner_lv_image, 
#                               inner_lv_dist_map], axis=1)
#         cv2.imwrite("vis.jpg", vis)
#         exit(0)
        
        return augmented_image, mask

    def __convert_mask_to_binary(self, mask):
        r"""Convert [1..255] mask to 0-1 binary mask"""
        binary_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        binary_mask[mask[..., 0] >= 128] = 1
        return binary_mask

    def __normalize_inner_lv_patch(self, inner_lv_image, inner_lv_mask, image, binary_mask, binary_inner_lv_mask):
        mask_ys, mask_xs = np.where(binary_mask > 0)
        inner_lv_mask_ys, inner_lv_mask_xs = np.where(binary_inner_lv_mask > 0)

        # resize so that inner lv mask is equal to mask in height and width
        mask_size = mask_xs.max() - mask_xs.min(), mask_ys.max() - mask_ys.min()
        inner_lv_mask_size = inner_lv_mask_xs.max() - inner_lv_mask_xs.min(), \
                            inner_lv_mask_ys.max() - inner_lv_mask_ys.min()
        inner_lv_mask_scale = mask_size[0] / inner_lv_mask_size[0], mask_size[1] / inner_lv_mask_size[1]
        inner_lv_image = cv2.resize(inner_lv_image, (int(inner_lv_mask_scale[0] * binary_inner_lv_mask.shape[1]), int(inner_lv_mask_scale[1] * binary_inner_lv_mask.shape[0])))
        inner_lv_mask = cv2.resize(inner_lv_mask, (int(inner_lv_mask_scale[0] * binary_inner_lv_mask.shape[1]), int(inner_lv_mask_scale[1] * binary_inner_lv_mask.shape[0])))

        # translate so that inner lv mask center coincide with mask center
        inner_lv_mask_xs = (inner_lv_mask_xs * inner_lv_mask_scale[0]).astype(int)
        inner_lv_mask_ys = (inner_lv_mask_ys * inner_lv_mask_scale[1]).astype(int)
        mask_center = np.array([np.mean(mask_xs), np.mean(mask_ys)]).astype(int)
        inner_lv_mask_center = np.array([np.mean(inner_lv_mask_xs), np.mean(inner_lv_mask_ys)]).astype(int)

        translation = mask_center - inner_lv_mask_center
        T = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        translated_inner_lv_image = cv2.warpAffine(inner_lv_image, T, (inner_lv_image.shape[1], inner_lv_image.shape[0]))
        cropped_inner_lv_image = self.__crop_and_pad(translated_inner_lv_image, (image.shape[1], image.shape[0]))
        translated_inner_lv_mask = cv2.warpAffine(inner_lv_mask, T, (inner_lv_mask.shape[1], inner_lv_mask.shape[0]))
        cropped_inner_lv_mask = self.__crop_and_pad(translated_inner_lv_mask, (image.shape[1], image.shape[0]))

        angle = np.random.uniform(low=-self.max_angle, high=self.max_angle)
        cropped_inner_lv_image = self.__rotate_image_around_center(cropped_inner_lv_image, mask_center, angle)
        cropped_inner_lv_mask = self.__rotate_image_around_center(cropped_inner_lv_mask, mask_center, angle)
        return cropped_inner_lv_image, cropped_inner_lv_mask
    
    def __crop_and_pad(self, image, target_size):
        r"""Extract a patch of target size whose top-left corner is (0, 0)"""
        h, w, _ = image.shape
        if h < target_size[1]:
            image = np.pad(image, ((0, target_size[1] - h), (0, 0), (0, 0)), mode="constant")
        if w < target_size[0]:
            image = np.pad(image, ((0, 0), (0, target_size[0] - w), (0, 0)), mode="constant")
        return image[:target_size[1], :target_size[0]]

    def __rotate_image_around_center(self, image, center, angle):
        rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    def __get_inner_lv_dist_map(self, inner_lv_mask):
        dist_map = self.__get_hausdorff_distance(inner_lv_mask)
        dist_map = np.clip(dist_map * self.dist_scale, 0, 1)
        return dist_map
    
    def __get_hausdorff_distance(self, cv2_msk):
        '''compute Hausdorff distance weights'''
        # 1. compute mask
        bw = cv2.cvtColor(cv2_msk, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 2. compute distance of mask's inner points
        dist_inner = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
        cv2.normalize(dist_inner, dist_inner, 0, 1.0, cv2.NORM_MINMAX)
        return np.float32(dist_inner)