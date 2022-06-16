import numpy as np
import cv2

from inference.config import DEFAULT


class CoarseContourExtractor(object):
    def __init__(self, smooth_filter=DEFAULT.coarse_contour_smooth_filter,
                 threshold=DEFAULT.coarse_contour_threshold):
        r"""

        Args:
            smooth_filter (array-like): Filter used to smooth contour, i.e. a 
                np.array of shape (2*smooth_width+1, )
            threshold (float): Threshold used to binarize LV mask prior to contour 
                extraction
        """
        self.smooth_filter = smooth_filter
        self.smooth_width = (len(smooth_filter) - 1) // 2
        self.threshold = threshold

    def __call__(self, mask):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, self.threshold, 255, 0)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(thresh,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        largest_contour = max([contour for contour in contours],
                              key=lambda x: cv2.contourArea(x))
        largest_contour = self.__smooth_contour(largest_contour)
        return largest_contour

    def __smooth_contour(self, contour):
        if contour.shape[0] < 21:
            return contour

        padded_contour = np.pad(contour,
                                ((self.smooth_width, self.smooth_width), (0, 0), (0, 0)),
                                mode="wrap")
        contour[:, 0, 0] = np.convolve(padded_contour[:, 0, 0],
                                       self.smooth_filter, mode="valid")
        contour[:, 0, 1] = np.convolve(padded_contour[:, 0, 1],
                                       self.smooth_filter, mode="valid")
        contour = contour.astype(int)
        return contour
