import numpy as np
import cv2
import tqdm

import inference.utils as utils
from inference.geometrical_analyzer.contour_extender import ContourExtender
from inference.config import DEFAULT


class PivotTracker:
    r"""Pivot tracker used to track pivot points"""
    def __init__(self, 
                 kernel_size=DEFAULT.tracking_kernel_size,
                 velocity=DEFAULT.cell_velocity):
        r"""

        Args:
            kernel_size (tuple): The kernel size (w, h) used for point tracking. 
                The point will be the center of the kernel. Due to the requirement of symmetricity, the kernel size must be odd. This is a tuple of 
                two int.
            velocity (float): The velocity of pixels (in real units).
        """
        self.kernel_size = kernel_size
        self.velocity = velocity
        self.contour_extender = ContourExtender()

    def __call__(self, initial_pivots, frames, contours, metadata):
        r"""Track pivots.

        Args:
            initial_pivots (np.array): Initial locations of the tracked pivots. 
                This is a np.array of shape (n_pivots, 2), i.e. (x, y) coordinates.
            frames (list): List of BGR frames. Each frame is a np.array of shape 
                (h, w, 3).
            contours (list): List of corresponding LV contours. Each contour is a 
                np.array of shape (n_cnt_points, 1, 2), i.e. (x, y) coordinates.
            metadata (EasyDict): Metadata extracted from DICOM dataset
                .frame_time (float): Duration of a frame in seconds.
                .x_scale (float): Pixel-to-real x-scale
                .y_scale (float): Pixel-to-real y-scale
                .heart_rate (float): Heart rate.
                .window (int): Number of frames per heart cycle.
        Returns:
            list: List of the tracked pivots. Each element is a np.array of shape 
                (n_pivots, 2), i.e. (x, y) coordinates.
        """
        max_allowed_dist = metadata["frame_time"] * self.velocity/metadata["x_scale"]
        
        pivot_sequence = []
        for i, (frame, contour) in enumerate(zip(frames, contours)): 
            frame = self.__preprocess_frame(frame)
            if i == 0:
                pivots, prev_frame = initial_pivots, frame
            else:
                extended_contour = self.contour_extender(frame, contour)
                new_pivots = self.__track_pivots(
                    pivots, extended_contour, 
                    max_allowed_dist, prev_frame, frame
                )
                pivots, prev_frame = new_pivots, frame
            pivot_sequence.append(pivots)
        pivot_sequence = np.array(pivot_sequence)
        return pivot_sequence
    
    def __preprocess_frame(self, frame):
        r"""Preprocess input frame for tracking"""
        frame = frame.astype(np.float32)/255.0
        return frame
    
    def __track_pivots(self, pivots, extended_contour, 
                       max_allowed_dist, prev_frame, frame):
        r"""Track pivots by multi-angles"""
        candidate_points = self.__get_candidate_points_for_all_frames(
            pivots, extended_contour, max_allowed_dist
        )
        
        next_pivot_points = []
        for pivot_point, candidates in zip(pivots, candidate_points):
            template = utils.images.crop_around_center(prev_frame, pivot_point, 
                                                       self.kernel_size)
            next_pivot_point, score = self.__track_one_frame(
                template, candidates, 
                frame, self.kernel_size
            )
            next_pivot_points.append(next_pivot_point)
        next_pivot_points = np.array(next_pivot_points)
        return next_pivot_points

    def __get_candidate_points_for_all_frames(self, 
                                              pivots, 
                                              extended_contour, 
                                              max_allowed_dist):
        r"""Get candidate points used for template matching for all frames"""
        dist = np.sqrt(np.sum(
            (extended_contour[None, :, :] - pivots[:, None, :])**2, 
            axis=-1
        ))
        max_allowed_dist = np.clip(np.min(dist, axis=-1), max_allowed_dist, None)
        candidate_points = [
            self.__get_candidate_points(pivot_point, extended_contour, max_dist) 
            for pivot_point, max_dist in zip(pivots, max_allowed_dist)
        ]
        return candidate_points

    def __get_candidate_points(self, pivot_point, extended_contour, max_allowed_dist):
        r"""Get candidate points used for template matching for one frame"""
        dist = np.sqrt(np.sum((extended_contour - pivot_point)**2, axis=-1))
        max_allowed_dist = max(np.min(dist), max_allowed_dist)
        extended_contour = extended_contour[dist <= max_allowed_dist]
        return extended_contour

    def __track_one_frame(self, template, candidates, 
                                        frame, kernel_size):
        r"""Rotate candidates and apply template matching to track pivots"""
        candidate_patches = np.array([
            utils.images.crop_around_center(frame, pt, kernel_size) 
            for pt in candidates
        ])
        score = self.__normalized_corr(template, candidate_patches)
        next_pivot_point = candidates[np.argmax(score)]
        score = np.max(score)
        return next_pivot_point, score
    
    def __normalized_corr(self, template, candidates):
        r"""Normalized correlation coefficient used for template matching"""
        corr = np.sum(template[None, ...] * candidates, axis=(1, 2, 3))
        l2_product = np.sqrt(np.sum(template**2)\
                           * np.sum(candidates**2, axis=(1, 2, 3)))
        score = corr / l2_product
        return score
