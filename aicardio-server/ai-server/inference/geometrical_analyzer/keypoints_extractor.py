import numpy as np
import cv2

from inference.config import DEFAULT
from inference.utils import geometry


class KeypointsExtractor(object):
    def __init__(self, 
                 vertical_threshold=DEFAULT.basepoint_vertical_thresold,
                 horizontal_threshold=DEFAULT.basepoint_horizontal_threshold,
                 max_step=DEFAULT.basepoint_max_step):
        self.vertical_threshold = vertical_threshold
        self.horizontal_threshold = horizontal_threshold
        self.max_step = max_step

    def __call__(self, contour):
        peak = self.__get_peak_point(contour)
        basepoints = self.__get_base_points(contour)
        return peak, basepoints
    
    def __get_peak_point(self, contour):
        ret, triangle = cv2.minEnclosingTriangle(contour)
        projections = self.__get_projection_on_contour(triangle, contour)
        peak = projections[np.argmin(projections[:, 0, 1]), 0]
        return peak
    
    def __get_base_points(self, contour):
        rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rect)
        sorted_box_points = np.array(sorted(box_points, key=lambda x: -x[1]))

        projections = self.__get_projection_on_contour(
            sorted_box_points[:, None, :], contour
        )
        projections = projections[np.argsort(projections[:, 0, 1])]
        
        basepoints = np.array(sorted(
            projections[2:, 0], 
            key=lambda x: x[0]
        ))
        basepoints = self.__finetune_base_points(
            basepoints, sorted_box_points, contour
        )
        return basepoints
    
    def __get_projection_on_contour(self, points, contour):
        dist = np.sum((contour[:, None, :, :] - points[None, :, :, :])**2,
                      axis=(-1, -2))
        projections = contour[np.argmin(dist, axis=0)]
        return projections
    
    def __finetune_base_points(self, basepoints, box_points, contour):
        A, B, C = geometry.line_eq(box_points[2], box_points[3])
        original_left_idx = np.where((contour == basepoints[0]).all(axis=-1))[0][0]
        original_right_idx = np.where((contour == basepoints[1]).all(axis=-1))[0][0]
        contour = contour.squeeze(1)

        vertical_threshold = self.__get_vertical_threshold(box_points)
        horizontal_threshold = self.__get_horizontal_threshold(
            box_points,
            contour[original_left_idx],
            contour[original_right_idx]
        )

        left_idx, right_idx = original_left_idx, original_right_idx
        while True:
            is_updated = False
            if left_idx+1 <= right_idx and left_idx+1 <= original_left_idx + self.max_step and\
                geometry.dist_to_line(contour[left_idx+1], (A, B, C)) >= vertical_threshold and\
                geometry.scaled_length(contour[left_idx+1], contour[right_idx]) >= horizontal_threshold:
                left_idx += 1
                is_updated = True
            if right_idx-1 >= left_idx and right_idx-1 >= original_right_idx - self.max_step and\
                geometry.dist_to_line(contour[right_idx-1], (A, B, C)) >= vertical_threshold and\
                geometry.scaled_length(contour[left_idx], contour[right_idx-1], 1, 1) >= horizontal_threshold:
                right_idx -= 1
                is_updated = True
            if not is_updated:
                break
        
        return contour[left_idx][:2], contour[right_idx][:2]
    
    def __get_vertical_threshold(self, box):
        vertical_length = min(geometry.scaled_length(box[0], box[-1]), 
                              geometry.scaled_length(box[0], box[-2]))
        vertical_threshold = self.vertical_threshold * vertical_length
        return vertical_threshold
    
    def __get_horizontal_threshold(self, box, left_base, right_base):
        horizontal_length = geometry.scaled_length(box[0], box[1])\
                          / geometry.cosine(box[0], box[1], left_base, right_base)
        horizontal_threshold = self.horizontal_threshold * horizontal_length
        return horizontal_threshold
