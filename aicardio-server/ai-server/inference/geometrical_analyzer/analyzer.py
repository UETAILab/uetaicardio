import time
import numpy as np
import cv2

from inference.interfaces import InferenceStep
from inference.geometrical_analyzer.coarse_contour_extractor import CoarseContourExtractor
from inference.geometrical_analyzer.keypoints_extractor import KeypointsExtractor
from inference.geometrical_analyzer.pivot_sampler import PivotSampler
from inference.geometrical_analyzer.kalman_points_smoother import KalmanPointsSmoother
from inference.config import DEFAULT


__all__ = ["CoarseGeometricalAnalyzer", "FineGeometricalAnalyzer"]


class CoarseGeometricalAnalyzer(InferenceStep):
    def __init__(self, logger=DEFAULT.logger):
        super(CoarseGeometricalAnalyzer, self).__init__("CoarseGeometricalAnalyzer",
                                                        logger)
        self.coarse_contour_extractor = CoarseContourExtractor()
        self.keypoints_extractor = KeypointsExtractor()
    
    def _validate_inputs(self, item):
        if "masks" not in item:
            item["is_valid"] = False
            self.logger.error("No mask to process")

    def _process(self, item):
        masks = item["masks"]
        contours, keypoints = self.run(masks)
        item.update({"contours": contours, "keypoints": keypoints})
        return item

    def run(self, masks):
        r"""Get coarse contours and keypoints

        This is the public method used for quick testing purpose

        Args:
            masks (list(np.array)): List of masks. This is a list of np.arrays, 
                each of which has shape (h, w, 3)
        Returns:
            list(np.array): List of np.array, each of which has shape (n_points, 1, 2)
                i.e. (x, y) absolute coordinates of contour points
            list(tuple): A tuple of two elements
                .The first element is a np.array of shape (2, ), i.e. (x, y)
                    absolute coordinates of LV peak
                .The second element is a tuple of two np.arrays, each of which
                    has shape (2, ), i.e. (x, y) absolute coordinates of left
                    and right LV base points
        """
        contours = [self.coarse_contour_extractor(mask) for mask in masks]
        keypoints = [self.keypoints_extractor(contour) for contour in contours]
        return contours, keypoints
    
    def _validate_outputs(self, item):
        if "contours" not in item:
            item["is_valid"] = False
            self.logger.error("No contours returned")
        if "keypoints" not in item:
            item["is_valid"] = False
            self.logger.error("No keypoint returned")

    def _visualize(self, item):
        if "visualize" not in item:
            item["visualize"] = {}
        item["visualize"][self.name] = []
        for frame, mask, contour, kps in zip(item["frames"], item["masks"], 
                                             item["contours"], item["keypoints"]):
            refined_mask = np.zeros_like(frame)
            cv2.fillPoly(refined_mask, contour[None, :, 0, :], (255, 255, 255))
            mask[..., 1:] = 0
            mask = cv2.addWeighted(refined_mask, 0.5, mask, 0.5, 0)
            mask = cv2.addWeighted(frame, 1, mask, 0.75, 0)
            
            peak, basepoints = kps
            cv2.circle(mask, tuple(peak), 3, (0, 255, 0), 3)
            cv2.circle(mask, tuple(basepoints[0]), 3, (0, 255, 0), 3)
            cv2.circle(mask, tuple(basepoints[1]), 3, (0, 255, 0), 3)
            
            item["visualize"][self.name].append(mask)
        return item
    
    def _log(self, item):
        pass


class FineGeometricalAnalyzer(InferenceStep):
    def __init__(self, 
                 n_contour_points_per_side=DEFAULT.n_contour_points_per_side,
                 smooth_filter=DEFAULT.fine_contour_smooth_filter,
                 n_smooth_iter=DEFAULT.fine_contour_n_smooth_iter,
                 logger=DEFAULT.logger):
        super(FineGeometricalAnalyzer, self).__init__("FineGeometricalAnalyzer",
                                                      logger)
        self.n_contour_points_per_side = n_contour_points_per_side
        self.keypoints_extractor = KeypointsExtractor()
        self.pivot_sampler = PivotSampler()
        self.kalman_points_smoother = KalmanPointsSmoother(
            covariance_scale=10, n_smooth_iter=n_smooth_iter
        )
        self.contour_extractor = CoarseContourExtractor(
            smooth_filter=smooth_filter
        )
    
    def _validate_inputs(self, item):
        if "contours" not in item:
            item["is_valid"] = False
            self.logger.error("No contour to process")
        if "keypoints" not in item:
            item["is_valid"] = False
            self.logger.error("No keypoint to process")

    def _process(self, item):
        frames = item["frames"]
        contours, keypoints = item["contours"], item["keypoints"]
        finetuned_contours, finetuned_keypoints = self.run(
            frames, contours, keypoints
        )
        item["finetuned_contours"] = finetuned_contours
        item["finetuned_keypoints"] = finetuned_keypoints
        return item

    def run(self, frames, contours, keypoints):
        r"""Fine-tune coarsely extracted LV contours and keypoints

        This is the public method used for quick testing purpose

        Args:
            frames (np.array): Extracted BGR frames from DICOM dataset. This is 
                a np.array of shape (n_frames, h, w, 3).
            contours (list(np.array)): List of np.array, each of which has shape 
                (n_points, 1, 2), i.e. (x, y) absolute coordinates of contour 
                points
            keypoints (list(tuple)): A tuple of two elements
                .The first element is a np.array of shape (2, ), i.e. (x, y)
                    absolute coordinates of LV peak
                .The second element is a tuple of two np.arrays, each of which
                    has shape (2, ), i.e. (x, y) absolute coordinates of left
                    and right LV base points
        Returns:
            list(np.array): List of np.array, each of which has shape 
                (n_points, 1, 2), i.e. (x, y) absolute coordinates of contour 
                points
            list(tuple): A tuple of two elements
                .The first element is a np.array of shape (2, ), i.e. (x, y)
                    absolute coordinates of LV peak
                .The second element is a tuple of two np.arrays, each of which
                    has shape (2, ), i.e. (x, y) absolute coordinates of left
                    and right LV base points
        """
        finetuned_contours = self.__finetune_contours(contours, keypoints)
        finetuned_contours = self.__smooth_contour_spatially(
            finetuned_contours, frames
        )
        finetuned_keypoints = [self.keypoints_extractor(contour) 
                               for contour in finetuned_contours]
        return finetuned_contours, finetuned_keypoints
    
    def __finetune_contours(self, contours, keypoints):
        sampled_contour_points = [self.pivot_sampler(contour, kps, 
                                             self.n_contour_points_per_side)
                                  for contour, kps in zip(contours, keypoints)]
        finetuned_contour_points = self.kalman_points_smoother(sampled_contour_points)
        finetuned_contours = [self.__convert_points_to_contours(points) 
                              for points in finetuned_contour_points]
        return finetuned_contours
    
    def __convert_points_to_contours(self, points):
        n_points = len(points)
        left_side = points[n_points//2::-1]
        right_side = points[:n_points//2:-1]
        contours = np.concatenate([left_side, right_side], axis=0)
        return contours[:, None, :]
    
    def __smooth_contour_spatially(self, contours, frames):
        smoothed_contours = []
        for frame, contour in zip(frames, contours):
            mask = np.zeros_like(frame)
            cv2.fillPoly(mask, contour[None, :, 0, :], (255, 255, 255))
            smoothed_contour = self.contour_extractor(mask)
            smoothed_contours.append(smoothed_contour)
        return smoothed_contours
    
    def _validate_outputs(self, item):
        if "finetuned_contours" not in item:
            item["is_valid"] = False
            self.logger.error("No contours returned")
        if "finetuned_keypoints" not in item:
            item["is_valid"] = False
            self.logger.error("No keypoint returned")

    def _visualize(self, item):
        if "visualize" not in item:
            item["visualize"] = {}
        item["visualize"][self.name] = []
        for (frame, mask, finetuned_contour, finetuned_kps) in zip(
            item["frames"], item["masks"], 
            item["finetuned_contours"], item["finetuned_keypoints"]
        ):
            finetuned_contour_mask = np.zeros_like(frame)
            cv2.fillPoly(finetuned_contour_mask, 
                         finetuned_contour[None, :, 0, :], 
                         (255, 255, 255))
            vis = cv2.addWeighted(frame, 1, finetuned_contour_mask, 0.5, 0)
            item["visualize"][self.name].append(vis)
        return item
    
    def _log(self, item):
        pass
