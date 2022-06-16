import numpy as np
import cv2

from inference.interfaces import InferenceStep
from inference.geometrical_analyzer.pivot_sampler import PivotSampler
from inference.geometrical_analyzer.pivot_reinitializer import PivotReinitializer
from inference.geometrical_analyzer.pivot_tracker import PivotTracker
from inference.geometrical_analyzer.kalman_points_smoother import KalmanPointsSmoother
from inference.config import DEFAULT


class PivotExtractor(InferenceStep):
    def __init__(self, config, logger=DEFAULT.logger):
        super(PivotExtractor, self).__init__("PivotExtractor",
                                             logger)
        self.config = config
        self.pivot_sampler = PivotSampler()
        self.pivot_reinitializer = PivotReinitializer()
        self.pivot_tracker = PivotTracker()
        self.kalman_points_smoother = KalmanPointsSmoother(
            covariance_scale=3, n_smooth_iter=10
        )
        
    def _validate_inputs(self, item):
        if "finetuned_contours" not in item:
            item["is_valid"] = False
            self.logger.error("No contour to process")
        if "finetuned_keypoints" not in item:
            item["is_valid"] = False
            self.logger.error("No keypoint to process")
    
    def _process(self, item):
        frames = item["frames"]
        finetuned_contours = item["finetuned_contours"]
        finetuned_keypoints = item["finetuned_keypoints"]
        metadata = item["metadata"]
        (initial_pivot_sequence, 
         tracked_pivot_sequence, 
         smoothed_pivot_sequence) = self.run(
             frames, metadata, finetuned_contours, finetuned_keypoints
        )
        item["tmp:initial_pivot_sequence"] = initial_pivot_sequence 
        item["tmp:tracked_pivot_sequence"] = tracked_pivot_sequence 
        item["pivot_sequence"] = smoothed_pivot_sequence
        return item
        

    def run(self, frames, metadata, finetuned_contours, finetuned_keypoints):
        r"""Extract pivots

        This is the public method used for quick testing purpose

        Args:
            frames (np.array): Extracted BGR frames from DICOM dataset. This is 
                a np.array of shape (n_frames, h, w, 3).
            metadata (dict): Metadata extracted from DICOM dataset, which 
                consists of
                .frame_time (float): Duration of a frame in seconds.
                .x_scale (float): Pixel-to-real x-scale
                .y_scale (float): Pixel-to-real y-scale
            finetuned_contours (list(np.array)): List of np.array, each of which 
                has shape (n_points, 1, 2), i.e. (x, y) absolute coordinates of 
                contour points
            finetuned_keypoints (list(tuple)): A tuple of two elements
                .The first element is a np.array of shape (2, ), i.e. (x, y)
                    absolute coordinates of LV peak
                .The second element is a tuple of two np.arrays, each of which
                    has shape (2, ), i.e. (x, y) absolute coordinates of left
                    and right LV base points
        Returns:
            np.array: (x, y) absolute coordinates of coarsely extracted pivot 
                points over frames. This is a np.array of shape 
                (n_frames, n_points, 2)
            np.array: (x, y) absolute coordinates of tracked pivot points over 
                frames. This is a np.array of shape (n_frames, n_points, 2)
            np.array: (x, y) absolute coordinates of Kalman-filter-smoothed 
                pivot points over frames. This is a np.array of shape 
                (n_frames, n_points, 2)
        """
        pivot_sequence = np.array([
            self.pivot_sampler(contour, kps, 5)
            for contour, kps in zip(finetuned_contours, 
                                    finetuned_keypoints)
        ])
        initial_pivot_sequence = pivot_sequence.copy()
        initial_pivots = self.pivot_reinitializer(
            pivot_sequence[0], frames, 
            self.config.reinitialized_pivots
        )
        tracked_pivot_sequence = self.__track_pivots(
            pivot_sequence, initial_pivots,
            frames, finetuned_contours, metadata
        )
        smoothed_pivot_sequence = self.kalman_points_smoother(tracked_pivot_sequence)
        return initial_pivot_sequence, tracked_pivot_sequence, smoothed_pivot_sequence
    
    def __track_pivots(self, pivot_sequence, initial_pivots, frames, contours, 
                       metadata):
        tracked_pivot_sequence = self.pivot_tracker(
            initial_pivots, frames,
            contours, metadata
        )
        tracked_pivot_sequence = tracked_pivot_sequence[:, self.config.tracked_pivots, :]
        pivot_sequence[:, self.config.tracked_pivots, :] = tracked_pivot_sequence
        return pivot_sequence
        
        
    def _validate_outputs(self, item):
        if "pivot_sequence" not in item:
            item["is_valid"] = False
            self.logger.error("No pivot sequence returned")
        
    def _visualize(self, item):
        if "visualize" not in item:
            item["visualize"] = {}
        item["visualize"][self.name] = []
        for i, (frame, mask, contour, initial_pivots,
                tracked_pivots, pivots) in enumerate(zip(
            item["frames"], item["masks"],
            item["finetuned_contours"],
            item["tmp:initial_pivot_sequence"],
            item["tmp:tracked_pivot_sequence"],
            item["pivot_sequence"]
        )):
            refined_mask_1 = np.zeros_like(frame)
            cv2.fillPoly(refined_mask_1, contour[None, :, 0, :], (255, 255, 255))
            for pivot in initial_pivots:
                cv2.circle(refined_mask_1, tuple(pivot), 3, (0, 255, 0), 3)
            for pivot in tracked_pivots:
                cv2.circle(refined_mask_1, tuple(pivot), 3, (255, 255, 0, 128), 3)
            refined_mask_1 = cv2.addWeighted(frame, 1, refined_mask_1, 0.5, 0)
                
            refined_mask_2 = frame.copy()
            for pivot in pivots:
                cv2.circle(refined_mask_2, tuple(pivot), 2, (255, 255, 0, 128), 2)
            
            mask = np.concatenate([refined_mask_1, refined_mask_2], axis=1)
            item["visualize"][self.name].append(mask)
        return item
    
    def _log(self, item):
        diff1 = np.sum((item["tmp:initial_pivot_sequence"] - item["tmp:tracked_pivot_sequence"])**2)
        diff2 = np.sum((item["tmp:tracked_pivot_sequence"] - item["pivot_sequence"])**2)
        self.logger.info(f"\n\tDifference between init and tracked pivots {diff1:.4f}\n\tDifference between tracked pivots and smoothed pivots {diff2:.4f}")
