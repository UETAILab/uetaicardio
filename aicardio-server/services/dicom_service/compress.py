# coding=utf-8
import os
import logging
import pydicom
import numpy as np
from PIL import Image

_logger = logging.getLogger(__name__)


class Compressor():
    def __init__(self, path, filename):
        """ Read dicom file to get fixel_array
        """
        self.path = path
        self.filename = filename
        try:
            self.dicom = pydicom.dcmread(
                os.path.join(self.path, self.filename))
            self.pixel_array = self.dicom.pixel_array
            # if dicom is a series of images
            self.is_series = len(self.pixel_array.shape) == 4
        except Exception as e:
            _logger.error(e)
            raise e

    def toJPEG(self):
        """compress dicom into jpeg images and return the path of folder contained them
        """
        if not os.path.exists("{}/{}_jpeg".format(self.path, self.filename)):
            os.makedirs("{}/{}_jpeg".format(self.path, self.filename))
        saved_imgs_path = []
        for idx, pixel_numpy in enumerate(self.pixel_array):
            img = Image.fromarray(pixel_numpy)
            img_path = "{}/{}_jpeg/{}_{}.jpg".format(self.path,
                                                self.filename, self.filename, idx)
            img.save(img_path)
            saved_imgs_path.append(img_path)
        return saved_imgs_path
