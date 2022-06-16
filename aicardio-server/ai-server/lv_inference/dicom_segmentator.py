import time
import numpy as np
import cv2
import importlib
import itertools

import torch
from torchvision import transforms

from lv_inference.lv_detection import YOLOv3LVDetector


class DICOMSegmentator:
    def __init__(self, args):
        r"""DICOM segmentator
        
        Args:
            args (EasyDict): Configuration of segmentator
                .detector_model_def (str): The .cfg file storing the config of the detector
                .detector_image_size (int): The size of the detector input
                .detector_model_weights (str): Path to detector .pth weight file
                .detector_batch_size (int): The batch size used in YOLOv3 LV detector
                .full_image_module_name (str): Path to the module containing the full-image segmentator class
                .full_image_segmentation_class (str): Name of the segmentator class
                .full_image_segmentation_weights (str): The .pth file containing the weights of the full-image segmentator
                .cropped_image_module_name (str): Path to the module containing the cropped segmentator class
                .cropped_image_segmentation_class (str): Name of the segmentator class
                .cropped_image_segmentation_weights (str): The .pth file containing the weights of the cropped segmentator
                .device (str): Device to run the DICOM segmentator
                .segmentation_image_size (int): Input size of the segmentators
        """
        self.config = args
        self.detector = self.__load_detector(args.detector_model_def,
                                             args.detector_image_size,
                                             args.detector_model_weights,
                                             args.device)
        self.full_image_segmentator = self.__load_segmentator(
            args.full_image_module_name, 
            args.full_image_segmentation_class, 
            args.full_image_segmentation_weights,
            args.device
        )
        self.cropped_image_segmentator = self.__load_segmentator(
            args.cropped_image_module_name, 
            args.cropped_image_segmentation_class, 
            args.cropped_image_segmentation_weights,
            args.device
        )
        self.segmentation_image_size = args.segmentation_image_size
        self.to_tensor = transforms.ToTensor()
    
    def __load_detector(self, model_def, image_size, model_weights, device="cuda:0"):
        return YOLOv3LVDetector(model_def, 
                                image_size, 
                                model_weights, 
                                device)
    
    def __load_segmentator(self, module_name, model_class, model_weights, device="cuda:0"):
        module = importlib.import_module(module_name)
        model = getattr(module, model_class)().to(device)
        checkpoint_data = torch.load(model_weights, map_location="cuda:0")
        model.load_state_dict(checkpoint_data["model"])
        model.eval()
        return model

    def get_segmentation_masks(self, dataset):
        msks = []
        start = time.time()
        all_bboxes = self.__get_all_bboxes(dataset)
        bboxes01 = self.__get_max_bbox(all_bboxes)
        if bboxes01 is not None:
            bboxes = self.__rescale_bboxes(dataset[0].image.shape, bboxes01)
        print(f"Detect time: {time.time()-start:.4f}")
        
        start = time.time()
        for i, item in enumerate(dataset):
            if bboxes01 is None:
                msk = self.__segment_lv(self.full_image_segmentator, item.image)
            else:
                cropped_img = self.__crop_bboxes(item.image[None, ...], bboxes)[0]
                cropped_msk = self.__segment_lv(self.cropped_image_segmentator, cropped_img)
                msk = np.zeros((item.h, item.w, 3), dtype=np.uint8)
                msk[bboxes[0][0, 1]:bboxes[0][0, 3], bboxes[0][0, 0]:bboxes[0][0, 2]] = cropped_msk
            msks.append(msk)
        print(f"Segment time: {time.time() - start}")
        return msks

    def __get_all_bboxes(self, dataset):
        images = np.concatenate([item.image[None, ...] for item in dataset], axis=0)
        all_bboxes = [self.detector.detect(images[i:i+self.config.detector_batch_size]) for i in range(0, len(images), self.config.detector_batch_size)]
        all_bboxes = list(itertools.chain.from_iterable(all_bboxes))
        return all_bboxes

    def __get_max_bbox(self, all_bboxes):
        all_bboxes = [x for x in all_bboxes if x.size > 0 and len(x) == 1]
        if len(all_bboxes) == 0:
            return None
        bboxes = [all_bboxes[0].copy()]
        bboxes[0][..., 0] = min([x[0][...,0] for x in all_bboxes])
        bboxes[0][..., 1] = min([x[0][...,1] for x in all_bboxes])
        bboxes[0][..., 2] = max([x[0][...,2] for x in all_bboxes])
        bboxes[0][..., 3] = max([x[0][...,3] for x in all_bboxes])
        return bboxes
    
    def __rescale_bboxes(self, image_shape, bboxes):
        for i in range(len(bboxes)):
            bboxes[i][:, [0, 2]] *= image_shape[1]
            bboxes[i][:, [1, 3]] *= image_shape[0]
            bboxes[i] = bboxes[i].astype(int)
        return bboxes
    
    def __crop_bboxes(self, images, bboxes):
        cropped_images = [image[boxes[0, 1]:boxes[0, 3], boxes[0, 0]:boxes[0, 2]] for image, boxes in zip(images, bboxes)]
        return cropped_images
    
    def __segment_lv(self, model, img):
        orig_shape = img.shape[1], img.shape[0]
        img = cv2.resize(img, (self.segmentation_image_size, self.segmentation_image_size), interpolation=cv2.INTER_CUBIC)
        img = self.to_tensor(img).to(self.config.device)
        with torch.no_grad():
            msk = model(img[None, ...])[0]
            msk = torch.sigmoid(msk)
        msk = msk.cpu().numpy().transpose((1, 2, 0))
        msk = np.uint8(msk * 255)
        msk = np.repeat(msk, 3, axis=-1)
        msk = cv2.resize(msk, orig_shape, interpolation=cv2.INTER_NEAREST)
        return msk