import cv2
import numpy as np
import tqdm

AVAILABLE_TRACKERS = {
    "Boosting": cv2.TrackerBoosting_create,
    "MIL": cv2.TrackerMIL_create,
    "KCF": cv2.TrackerKCF_create,
    "TLD": cv2.TrackerTLD_create,
    "MedianFlow": cv2.TrackerMedianFlow_create,
    "GOTURN": cv2.TrackerGOTURN_create,
    "MOSSE": cv2.TrackerMOSSE_create,
    "CSRT": cv2.TrackerCSRT_create,
}

class CV2PivotTracker:
    r"""Multi-angle pivot tracker used to track pivot points"""

    def __init__(self, tracker_config):
        r"""
        Args:
            tracker_config (EasyDict): The configuration used for tracking. This is an EasyDict consisting of
                .kernel_size (tuple): The kernel size (w, h) used for point tracking. The point will be the center of the kernel. Due to the requirement of symmetricity, the kernel size must be odd. This is a tuple of two int.
                .tracker_type (str): The type of tracker to be used.
        """
        if not self.__is_valid_config(tracker_config):
            raise ValueError("Invalid tracking configuration!")
        self.config = tracker_config
        self.tracker = cv2.MultiTracker_create()
        self.tracker_type = self.config.tracker_type
        self.kernel_size = self.config.kernel_size
    
    def __is_valid_config(self, config):
        r"""Check for tracking configuration validity"""
        is_valid = config.kernel_size[0]%2 == 1
        is_valid = is_valid and (config.kernel_size[1]%2 == 1)
        is_valid = is_valid and (config.tracker_type in AVAILABLE_TRACKERS)
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
        self.__initialize_tracker(initial_pivots, frames[0])
        pivot_sequence = []
        for i, frame in enumerate(frames):
            success, boxes = self.tracker.update(frame)
            if not success:
                raise InterruptedError("Tracker failed in tracking")
            pivots = self.__parse_tracker_outputs(boxes)
            pivot_sequence.append(pivots)
        return pivot_sequence
    
    def __initialize_tracker(self, initial_pivots, initial_frame):
        r"""Initialize tracker with initial pivots and frame"""
        for pivot in initial_pivots:
            self.tracker.add(AVAILABLE_TRACKERS[self.tracker_type](),
                             initial_frame,
                             self.__point_to_kernel(pivot))

    def __point_to_kernel(self, point):
        r"""Convert a (x, y) coordinate to (x, y, w, h) box"""
        x = point[0] - self.kernel_size[0]//2
        y = point[1] - self.kernel_size[1]//2
        w, h = self.kernel_size
        return x, y, w, h
    
    def __parse_tracker_outputs(self, boxes):
        r"""Parse the output of self.tracker.update(frame) to get (x, y) pivots"""
        pivots = np.array([(int(box[0] + box[2]/2),
                            int(box[1] + box[3]/2)) for box in boxes])
        return pivots
