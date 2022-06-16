import os
import time
import uuid
import cv2
import numpy as np
import tqdm


class MultiAnglePivotTracker:
    r"""Multi-angle pivot tracker used to track pivot points"""

    def __init__(self, tracker_config):
        r"""
        Args:
            tracker_config (EasyDict): The configuration used for tracking. This is an EasyDict consisting of
                .kernel_size (tuple): The kernel size (w, h) used for point tracking. The point will be the center of the kernel. Due to the requirement of symmetricity, the kernel size must be odd. This is a tuple of two int.
                .velocity (float): The velocity of pixels (in real units).
                .angles (list): List of angles to be used in multi-angle tracking. The angles are float numbers, which are angles in degrees, not radians.
        """
        if not self.__is_valid_config(tracker_config):
            raise ValueError("Invalid tracking configuration!")
        self.config = tracker_config
        self.kernel_size = self.config.kernel_size
        self.velocity = self.config.velocity
        self.angles = self.config.angles
        self.contour_extender = ContourExtender()
    
    def __is_valid_config(self, config):
        r"""Check for tracking configuration validity"""
        is_valid = config.kernel_size[0]%2 == 1
        is_valid = is_valid and (config.kernel_size[1]%2 == 1)
        return is_valid

    def track_pivots(self, initial_pivots, frames, masks, contours, metadata):
        r"""Track pivots.

        Args:
            initial_pivots (np.array): Initial locations of the tracked pivots. This is a np.array of shape (n_pivots, 2), i.e. (x, y) coordinates.
            frames (list): List of BGR frames. Each frame is a np.array of shape (h, w, 3).
            masks (list): List of corresponding BGR LV segmentation. Each mask is a np.array of shape (h, w, 3).
            contours (list): List of corresponding LV contours. Each contour is a np.array of shape (n_cnt_points, 1, 2), i.e. (x, y) coordinates.
            metadata (EasyDict): Metadata extracted from DICOM dataset, which consists of
                .frame_time (float): Duration of a frame in seconds.
                .x_scale (float): Pixel-to-real x-scale
                .y_scale (float): Pixel-to-real y-scale
                .heart_rate (float): Heart rate.
                .window (int): Number of frames per heart cycle.
        Returns:
            list: List of the tracked pivots. Each element is a np.array of shape (n_pivots, 2), i.e. (x, y) coordinates.
        """
        max_allowed_dist = metadata.frame_time * self.velocity/metadata.x_scale
        
        pivot_sequence = []
        progress_bar = tqdm.tqdm(enumerate(zip(frames, masks, contours)), total=len(frames))
        for i, (frame, mask, contour) in progress_bar:
            frame = self.__preprocess_frame(frame)
            if i == 0:
                pivots, prev_frame = initial_pivots, frame
            else:
                extended_contour = self.contour_extender.extend_contour(frame, contour)
                new_pivots = self.__track_pivots(pivots, extended_contour, max_allowed_dist, prev_frame, frame)
                pivots, prev_frame = new_pivots, frame
            pivot_sequence.append(pivots)
        return pivot_sequence
    
    def __preprocess_frame(self, frame):
        r"""Preprocess input frame for tracking"""
        frame = frame.astype(np.float32)/255.0
        return frame
    
    def __track_pivots(self, pivots, extended_contour, max_allowed_dist, prev_frame, frame):
        r"""Track pivots by multi-angles"""
        rotated_frames, rot_mats = self.__rotate_frame_by_angles(frame)
        candidate_points = self.__get_candidate_points_for_all_frames(pivots, extended_contour, max_allowed_dist)
        
        next_pivot_points = []
        for pivot_point, candidates in zip(pivots, candidate_points):
            template = self.__crop_and_normalize(prev_frame, pivot_point, self.kernel_size)
            next_pivot_point, score = self.__rotate_and_track_based_on_sim(template, candidates, rotated_frames, rot_mats, self.kernel_size)
            next_pivot_points.append(next_pivot_point)
        next_pivot_points = np.array(next_pivot_points)
        return next_pivot_points            
    
    def __rotate_frame_by_angles(self, frame):
        r"""Rotate frame by multiple angles and return both rotated frames and rotation matrices"""
        rotated_frames, rotation_matrices = [], []
        for angle in self.angles:
            rotated_frame, rotation_matrix = self.__rotate_image_around_center(frame, angle, True)
            rotated_frames.append(rotated_frame)
            rotation_matrices.append(rotation_matrix)
        return rotated_frames, rotation_matrices

    def __rotate_image_around_center(self, image, angle, return_rot_mat=False):
        r"""Rotate an image by some angle around its center"""
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        if return_rot_mat:
            return result, rot_mat
        else:
            return result

    def __get_candidate_points_for_all_frames(self, pivots, extended_contour, max_allowed_dist):
        r"""Get candidate points used for template matching for all frames"""
        dist = np.sqrt(np.sum((extended_contour[None, :, :] - pivots[:, None, :])**2, axis=-1)) # (n, m)
        max_allowed_dist = np.clip(np.min(dist, axis=-1), max_allowed_dist, None)
        candidate_points = [self.__get_candidate_points(pivot_point, extended_contour, max_dist) for pivot_point, max_dist in zip(pivots, max_allowed_dist)]
        return candidate_points

    def __get_candidate_points(self, pivot_point, extended_contour, max_allowed_dist):
        r"""Get candidate points used for template matching for one frame"""
        dist = np.sqrt(np.sum((extended_contour - pivot_point)**2, axis=-1))
        max_allowed_dist = max(np.min(dist), max_allowed_dist)
        extended_contour = extended_contour[dist <= max_allowed_dist]
        return extended_contour

    def __rotate_and_track_based_on_sim(self, template, candidates, rotated_frames, rot_mats, kernel_size):
        r"""Rotate candidates and apply template matching to track pivots"""
        next_pivot_points, scores = [], []
        for rotated_frame, rot_mat in zip(rotated_frames, rot_mats):
            rotated_candidates = self.__get_rotated_candidate_points(candidates, rot_mat).astype(int)
            rotated_candidates = np.array([self.__crop_and_normalize(rotated_frame, pt, kernel_size) for pt in rotated_candidates])
            score = self.__normalized_corr(template, rotated_candidates)
            next_pivot_point = candidates[np.argmax(score)]
            next_pivot_points.append(next_pivot_point)
            scores.append(np.max(score))
        best_score = np.max(scores)
        best_next_pivot_point = next_pivot_points[np.argmax(scores)]
        return best_next_pivot_point, best_score

    def __rotate_and_track_based_on_dist(self, template, candidates, rotated_frames, rot_mats, kernel_size):
        r"""Rotate candidates and apply template matching to track pivots"""
        next_pivot_points, scores = [], []
        for rotated_frame, rot_mat in zip(rotated_frames, rot_mats):
            rotated_candidates = self.__get_rotated_candidate_points(candidates, rot_mat).astype(int)
            rotated_candidates = np.array([self.__crop_and_normalize(rotated_frame, pt, kernel_size) for pt in rotated_candidates])
            score = self.__normalized_euclidean(template, rotated_candidates)
            next_pivot_point = candidates[np.argmin(score)]
            next_pivot_points.append(next_pivot_point)
            scores.append(np.min(score))
        best_score = np.min(scores)
        best_next_pivot_point = next_pivot_points[np.argmin(scores)]
        return best_next_pivot_point, best_score
    
    def __normalized_corr(self, template, candidates):
        r"""Normalized correlation coefficient used for template matching"""
        score = np.sum(template[None, ...] * candidates, axis=(1, 2, 3))
        score = score / np.sqrt(np.sum(template**2) * np.sum(candidates**2, axis=(1, 2, 3)))
        return score
    
    def __normalized_euclidean(self, template, candidates):
        r"""Normalized Euclidean distance used for template matching"""
        dist = np.sum((template[None, ...] - candidates)**2, axis=(1, 2, 3))
        dist = dist / (np.sum(template**2) + np.sum(candidates**2, axis=(1, 2, 3)) + 1e-10)
        return dist

    def __crop_and_normalize(self, image, center, box_size):
        r"""Crop a patch of size `box_size` around a point `center` from `image`"""
        x1, y1 = center[0] - (box_size[0] - 1)//2, center[1] - (box_size[1] - 1)//2
        x2, y2 = x1 + box_size[0], y1 + box_size[1]
        cropped = image[y1:y2, x1:x2]
#         cropped = (cropped - cropped.mean()) / cropped.std() # use for Euclidean distance
        return cropped

    def __get_rotated_candidate_points(self, candidate_points, rot_mat):
        r"""Rotate candidate points using a rotation matrix"""
        candidate_points = np.concatenate([candidate_points, np.ones((len(candidate_points), 1))], axis=-1)
        candidate_points = np.dot(candidate_points, rot_mat.T)
        return candidate_points


class ContourExtender:
    r"""Contour extender used in pivot tracking"""
    def __init__(self, dilation_rate=1.15, contraction_rate=0.85):
        r"""
        
        Args:
            dilation_rate (float): The rate at which contour is scaled up.
            contraction_rate (float): The rate at which contour is scaled down.
        """
        self.dilation_rate = dilation_rate
        self.contraction_rate = contraction_rate

    def extend_contour(self, frame, contour):
        r"""Contract and dilate LV contour.
        
        Args:
            frame (np.array): BGR frame. This is a np.array of shape (h, w, 3)
            contour (np.array): LV contour. This is a np.array of shape (n_cnt_points, 1, 2), i.e. (x, y) coordinates.
        Returns:
            np.array: Coordinates of contour neigbor pixels. This is a np.array of shape (n_neighbors, 2), i.e. (x, y) coordinates.
        """
        neighbor_map = self.__get_contour_neighbor_map(frame, contour)
        neighbor_ys, neighbor_xs = np.where(neighbor_map > 0)
        extended_contour = np.concatenate([neighbor_xs[..., None], neighbor_ys[..., None]], axis=-1)
        return extended_contour

    def __get_contour_neighbor_map(self, frame, contour):
        r"""Get binary mask with white pixels indicating contour neighbors"""
        neighbor_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        inner_contour = self.__scale_contour(contour, self.contraction_rate)
        outer_contour = self.__scale_contour(contour, self.dilation_rate)
        cv2.fillPoly(neighbor_map, pts=[outer_contour], color=(255, 255, 255))
        cv2.fillPoly(neighbor_map, pts=[inner_contour], color=(0, 0, 0))
        return neighbor_map

    def __scale_contour(self, contour, scale=1.0):
        r"""Scale LV contour curve"""
        cx, cy = self.__get_center(contour)
        contour = contour - [cx, cy]
        scaled_contour = contour * scale
        scaled_contour += [cx, cy]
        scaled_contour = scaled_contour.astype(np.int32)
        return scaled_contour

    def __get_center(self, contour):
        r"""Get center of the LV contour curve"""
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        return cX, cY