import numpy as np
import cv2
from inference.config import DEFAULT


class PivotReinitializer(object):
    def __init__(self, 
                 max_distance=DEFAULT.pivots_reinitialization_max_distance, 
                 score_threshold=DEFAULT.pivots_reinitialization_score_thresold):
        self.max_distance = max_distance
        self.score_threshold = score_threshold

    def __call__(self, coarse_initial_pivots, frames, reinitialized_pivots):
        r"""Initialize pivots for tracking"""
        initial_frame = self.__get_initial_frame(frames)
        
        finetuned_pivots = []
        for i, pivot in enumerate(coarse_initial_pivots):
            candidates = self.__get_candidate_pivots(pivot, initial_frame)
            if candidates is None or i not in reinitialized_pivots:
                finetuned_pivots.append(pivot)
            else:
                chosen_candidate = self.__choose_from_candiates(pivot, candidates)
                finetuned_pivots.append(chosen_candidate)
        finetuned_pivots = np.array(finetuned_pivots)
        return finetuned_pivots
    
    def __get_initial_frame(self, frames):
        initial_frame = self.__get_initial_foreground_frame(frames)
        initial_frame = self.__preprocess_initial_frame(initial_frame)
        return initial_frame

    def __get_initial_foreground_frame(self, frames):
        frames = np.array(frames)
        frame_mean = frames.mean(axis=0).astype(np.uint8)
        initial_frame = frames[0].copy()
        initial_frame[(initial_frame - frame_mean) == 0] = 0
        return initial_frame
    
    def __preprocess_initial_frame(self, frame):
        frame = cv2.medianBlur(frame, 3)
        frame = cv2.medianBlur(frame, 5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    def __get_candidate_pivots(self, pivot, initial_frame):
        roi_mask = np.zeros_like(initial_frame)
        cv2.circle(roi_mask, tuple(pivot), self.max_distance, (255, 255, 255), -1)
        candidates = cv2.goodFeaturesToTrack(
            initial_frame, mask=roi_mask, maxCorners=100,
            qualityLevel=self.score_threshold, minDistance=1, blockSize=61
        )
        return candidates
    
    def __choose_from_candiates(self, pivot, candidates):
        candidates = np.array(candidates).squeeze(1)
        dist = np.sum((candidates - pivot)**2, axis=-1)
        nearest_candidate = candidates[np.argmin(dist)].astype(int)
        return nearest_candidate
