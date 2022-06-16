import cv2
import time
import torch
import importlib
import itertools
import numpy as np

from easydict import EasyDict
from torchvision.transforms import ToTensor

from echols.log import logger
from echols.segnet import AuxUNet as SegNet
from echols.detector import YOLOv3LVDetector


class DICOMSegmentor:
    def __init__(self, ckpt, input_size=256, device='cpu', **kwargs):
        self.config = EasyDict(locals())
        self.model = self._load_segmentor()

    def predict_masks(self, dataset):
        start = time.time()
        masks = []
        for i, item in enumerate(dataset):
            mask = self._segment_lv(item['image'])
            masks.append(mask)
        logger.info(f"Segment time: {time.time() - start}")
        return masks

    def _segment_lv(self, img):
        orig_shape = img.shape[1], img.shape[0]
        img = cv2.resize(img, (self.config.input_size, self.config.input_size), interpolation=cv2.INTER_CUBIC)
        img = ToTensor()(img).to(self.config.device)
        with torch.no_grad():
            msk = self.model(img[None, ...])[0]
            msk = torch.sigmoid(msk)
        msk = msk.cpu().numpy().transpose((1, 2, 0))
        msk = np.uint8(msk * 255)
        msk = np.repeat(msk, 3, axis=-1)
        msk = cv2.resize(msk, orig_shape, interpolation=cv2.INTER_NEAREST)
        return msk

    def _load_segmentor(self):
        model = SegNet()
        if self.config.ckpt:
            ckpt_data = torch.load(self.config.ckpt, map_location="cpu")
            model.load_state_dict(ckpt_data["model"])
        model.to(self.config.device).eval()
        return model


class DICOMCropSegmentor(DICOMSegmentor):
    def __init__(self, detector, ckpt, input_size=256, device='cpu', **kwargs):
        super().__init__(ckpt, input_size, device, **kwargs)
        self.detector = detector

    def predict_masks(self, dataset):
        start = time.time()
        all_bboxes = self._get_all_bboxes(dataset)
        bboxes01 = self._get_max_bbox(all_bboxes)
        if bboxes01 is not None:
            bboxes = self._rescale_bboxes(dataset[0]['image'].shape, bboxes01)
        logger.info(f"Detect time: {time.time()-start:.4f}")

        start = time.time()
        masks = []
        for i, item in enumerate(dataset):
            if bboxes01 is not None:
                cropped_img = self._crop_bboxes(item['image'][None, ...], bboxes)[0]
                cropped_msk = self._segment_lv(cropped_img)
                mask = np.zeros((item['h'], item['w'], 3), dtype=np.uint8)
                mask[bboxes[0][0, 1]:bboxes[0][0, 3], bboxes[0][0, 0]:bboxes[0][0, 2]] = cropped_msk
            else:
                mask = self._segment_lv(item['image'])
            masks.append(mask)
        logger.info(f"Segment time: {time.time() - start}")
        return masks

    def _get_all_bboxes(self, dataset):
        images = np.concatenate([item['image'][None, ...] for item in dataset], axis=0)
        all_bboxes = [self.detector.detect(images[i:i+self.detector.batch_size]) for i in range(0, len(images), self.detector.batch_size)]
        all_bboxes = list(itertools.chain.from_iterable(all_bboxes))
        return all_bboxes

    def _get_max_bbox(self, all_bboxes):
        all_bboxes = [x for x in all_bboxes if x.size > 0 and len(x) == 1]
        if len(all_bboxes) == 0:
            return None
        bboxes = [all_bboxes[0].copy()]
        bboxes[0][..., 0] = min([x[0][...,0] for x in all_bboxes])
        bboxes[0][..., 1] = min([x[0][...,1] for x in all_bboxes])
        bboxes[0][..., 2] = max([x[0][...,2] for x in all_bboxes])
        bboxes[0][..., 3] = max([x[0][...,3] for x in all_bboxes])
        return bboxes

    def _rescale_bboxes(self, image_shape, bboxes):
        for i in range(len(bboxes)):
            bboxes[i][:, [0, 2]] *= image_shape[1]
            bboxes[i][:, [1, 3]] *= image_shape[0]
            bboxes[i] = bboxes[i].astype(int)
        return bboxes

    def _crop_bboxes(self, images, bboxes):
        cropped_images = [image[boxes[0, 1]:boxes[0, 3], boxes[0, 0]:boxes[0, 2]] for image, boxes in zip(images, bboxes)]
        return cropped_images


if __name__ == "__main__":
    img = cv2.imread('assets/dog.png')
    dataset = [{'image': img, 'h':256, 'w':256}]

    segmentor = DICOMSegmentor('ckpts/full_aux_giangunet_invhd_0009_0.8604_best.pth')
    segmentor.predict_masks(dataset)

    detector = YOLOv3LVDetector(config='echols/yolov3/config/yolov3-custom.cfg', ckpt='ckpts/0.7435_yolov3_ckpt_75.pth')
    segmentor = DICOMCropSegmentor(detector, 'ckpts/cropped_aux_bce_invhd_giangunet_0014_0.8642_best.pth')
    segmentor.predict_masks(dataset)
