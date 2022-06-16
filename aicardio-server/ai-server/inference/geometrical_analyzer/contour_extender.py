import numpy as np
import cv2
from inference.config import DEFAULT


class ContourExtender(object):
    r"""Contour extender used in pivot tracking"""
    def __init__(self, 
                 dilation_rate=DEFAULT.contour_dilation_rate, 
                 contraction_rate=DEFAULT.contour_contraction_rate):
        r"""
        
        Args:
            dilation_rate (float): The rate at which contour is scaled up.
            contraction_rate (float): The rate at which contour is scaled down.
        """
        self.dilation_rate = dilation_rate
        self.contraction_rate = contraction_rate

    def __call__(self, frame, contour):
        r"""Contract and dilate LV contour.
        
        Args:
            frame (np.array): BGR frame. This is a np.array of shape (h, w, 3)
            contour (np.array): LV contour. This is a np.array of shape 
                (n_cnt_points, 1, 2), i.e. (x, y) coordinates.
        Returns:
            np.array: Coordinates of contour neigbor pixels. This is a np.array of 
                shape (n_neighbors, 2), i.e. (x, y) coordinates.
        """
        neighbor_map = self.__get_contour_neighbor_map(frame, contour)
        neighbor_ys, neighbor_xs = np.where(neighbor_map > 0)
        extended_contour = np.concatenate([
            neighbor_xs[..., None], 
            neighbor_ys[..., None]
        ], axis=-1)
        return extended_contour

    def __get_contour_neighbor_map(self, frame, contour):
        r"""Get binary mask with white pixels indicating contour neighbors"""
        neighbor_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        neighbor_map = self.__scale_and_fill_polygon(
            neighbor_map, contour,
            self.contraction_rate, (0, 0, 0)
        )
        neighbor_map = self.__scale_and_fill_polygon(
            neighbor_map, contour,
            self.dilation_rate, (255, 255, 255)
        )
        return neighbor_map
    
    def __scale_and_fill_polygon(self, image, polygon, scale, color):
        polygon = self.__scale_polygon(polygon, scale)
        cv2.fillPoly(image, pts=[polygon], color=color)
        return image

    def __scale_polygon(self, polygon, scale=1.0):
        r"""Scale polygon"""
        cx, cy = self.__get_center(polygon)
        polygon = polygon - [cx, cy]
        scaled_polygon = polygon * scale
        scaled_polygon += [cx, cy]
        scaled_polygon = scaled_polygon.astype(np.int32)
        return scaled_polygon

    def __get_center(self, polygon):
        r"""Get center of a polygon"""
        M = cv2.moments(polygon)
        if M['m00'] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        return cX, cY
