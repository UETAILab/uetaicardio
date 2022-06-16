import os
import time
import math
import cv2
import numpy as np

from easydict import EasyDict
from pykalman import KalmanFilter

from .lvstats.pivot_analyzer import PivotAnalyzer
from .lvstats.multiangle_pivot_tracker import MultiAnglePivotTracker
from .lvstats.cv2_pivot_tracker import CV2PivotTracker
from .lvstats.ef_utils import *
from echols.log import logger


class EFCalculator:
    def __init__(self):
        # 2C
        tracker_config = EasyDict(dict(
            kernel_size=(91, 91), #(31, 31),#(121, 121), # (61, 61),
            velocity=7.2,
            angles=[0]
        ))
        self.pivot_tracker = MultiAnglePivotTracker(tracker_config)
        self.pivot_analyzer = PivotAnalyzer()
        self.kalman_params = EasyDict(dict(
            transition_matrix=[[1, 1, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 1],
                               [0, 0, 0, 1]],
            observation_matrix=[[1, 0, 0, 0],
                                [0, 0, 1, 0]]
        ))

    def _get_contours(self, masks):
        contours = [extract_contour(m) for m in masks]
        contours = [c for c in contours if len(c) > 0] #fixbug
        return contours

    def _get_all_peak_points(self, contours):
        peaks = [get_peak_by_middle_mfpunet(c) for c in contours]
        return peaks

    def _get_all_base_points(self, contours):
        basepoints = [adjusted_get_base_points_by_bbox_basepoints(c) for c in contours]
        return basepoints

    def _compute_all_areas(self, contours, metadata):
        areas = [compute_area(c, metadata['x_scale'], metadata['y_scale']) for c in contours]
        return areas

    def _compute_all_lv_lengths(self, contours, pivot_sequence, metadata):
        lv_lengths = [compute_lv_length(c, pivots[3], pivots[[0, -1]], metadata['x_scale'], metadata['y_scale']) for c, pivots in zip(contours, pivot_sequence)]
        return lv_lengths

    def _compute_all_volumes(self, areas, lv_lengths):
        volumes = np.array([ 8.0 / 3.0 / math.pi * area * area / l[0] \
                for area, l in zip(areas, lv_lengths)])
        return volumes

    def compute_ef(self, msks, metadata, dataset, output_dir=None):
        start = time.time()
        contours = self._get_contours(msks) #1
        peaks = self._get_all_peak_points(contours) #2
        basepoints = self._get_all_base_points(contours) #3
        contours, peaks, basepoints = self._smooth_contours(contours, peaks, basepoints) #4
        pivot_sequence = self._estimate_pivots(contours, peaks, basepoints, dataset, msks, metadata) #5
        areas = self._compute_all_areas(contours, metadata) #6
        lengths = self._compute_all_lv_lengths(contours, pivot_sequence, metadata) #7
        lv_lengths = lengths
        volumes = self._compute_all_volumes(areas, lv_lengths) #8
        logger.info(f"Start to volumes time: {time.time() - start:.4f}")

        # Compute EF and GLS
        EF, efs, idxEF = compute_EF(metadata['window'], volumes) #9
        GLS1 = get_gls_by_basepoints(metadata['window'], basepoints, contours) #10
        GLS2 = get_gls_by_segments(metadata['window'], pivot_sequence) #11

        visualize(output_dir, dataset, msks, contours, \
                pivot_sequence, basepoints, areas, lengths, volumes, self.pivot_tracker)
        plot(output_dir, areas, lengths, volumes, EF, efs, idxEF, GLS1, GLS2)

        results = {
            "contours": [],
            "pivot_sequence": [],
            "areas": [],
            "volumes": [],
            "idxEF": idxEF,
            "efs": efs,
            "x_width": msks[0].shape[1],
            "y_height": msks[0].shape[0],
        }
        for msk, contour, pivots, area, volume in zip(msks, contours, pivot_sequence, areas, volumes):
            #results["contours"].append([ {"x": point[0, 0] / msk.shape[1],
                                          #"y": point[0, 1] / msk.shape[0]}
                                    #for point in contour])
            results["contours"].append([[int(point[0, 0]), int(point[0, 1])] for point in contour])
            #results["pivot_sequence"].append([ {"x": point[0] / msk.shape[1],
                                                #"y": point[1] / msk.shape[0]}
                                    #for point in pivots])
            results["areas"].append(area)
            results["volumes"].append(volume)
        results["ef"] = EF
        results["basepoint_gls"] = GLS1
        results["segmental_gls"] = GLS2
        return results

    def _estimate_pivots(self, contours, peaks, basepoints, dataset, msks, metadata):
        frames = [data["image"] for data in dataset]
        detected_pivots_sequence = [self.pivot_analyzer.detect_pivots(c, p, b, n_segments_per_side=3) \
                                        for c, p, b in zip(contours, peaks, basepoints)]
        initial_pivots = init_pivots_for_tracking(
            detected_pivots_sequence[0],
            frames, [1, 2, 4, 5]
        )
        tracked_pivot_sequence = self.pivot_tracker.track_pivots(
            initial_pivots,
            frames, msks,
            contours, metadata
        )
        pivot_sequence = np.array([[tracked_pivots[0], tracked_pivots[1], tracked_pivots[2], tracked_pivots[3], \
                                    tracked_pivots[4], tracked_pivots[5], tracked_pivots[6]] \
                                for detected_pivots, tracked_pivots in \
                                zip(detected_pivots_sequence, tracked_pivot_sequence)]).astype(int)
        #pivot_sequence = np.array([tp for tp in tracked_pivot_sequence]).astype(int)
        #projected_pivot_sequence = np.array([project_pivots_on_contour(pivots, contour) \
                #for pivots, contour in zip(pivot_sequence, contours)])
        #pivot_sequence[:, [0, 3, 6]] = projected_pivot_sequence[:, [0, 3, 6]]
        #pivot_sequence = projected_pivot_sequence
        pivot_sequence = smooth_pivots(pivot_sequence, self.kalman_params, covariance_scale=5)
        return pivot_sequence

    def _smooth_contours(self, contours, peaks, basepoints):
        assert len(contours) == len(peaks) == len(basepoints), 'something went wrong'
        start = time.time()
        detected_pivots_sequence = [self.pivot_analyzer.detect_pivots(c, p, b, n_segments_per_side=20) \
                                        for c, p, b in zip(contours, peaks, basepoints)]
        pivot_sequence = np.concatenate([pivots[None, ...] for pivots in detected_pivots_sequence])
        pivot_sequence = smooth_pivots(pivot_sequence, self.kalman_params, n_iter=3)

        contours = [convert_pivots_to_contours(pivots) for pivots in pivot_sequence]
        peaks = [tuple(contour[0, 0]) for contour in contours]

        basepoints = [{"left_basepoint": tuple(contour[len(contour)//2, 0]), \
                       "right_basepoint": tuple(contour[len(contour)//2+1, 0]), \
                       "box": basepoint["box"]} \
                for contour, basepoint in zip(contours, basepoints)]

        logger.info(f"Contour smooth time: {time.time() - start}")
        return contours, peaks, basepoints
