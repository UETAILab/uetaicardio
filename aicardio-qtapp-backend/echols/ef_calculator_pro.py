import os
import time
import math
import cv2
import numpy as np

from easydict import EasyDict
from pykalman import KalmanFilter

from .lvstats.multiangle_pivot_tracker import MultiAnglePivotTracker
from .lvstats.cv2_pivot_tracker import CV2PivotTracker
from .lvstats.ef_utils import *
from .lvstats.pivot_extractor import PivotExtractor
from echols.log import logger


class EFCalculatorPro:
    def __init__(self):
        # 2C
        tracker_config = EasyDict(dict(
            kernel_size=(91, 91), #(31, 31),#(121, 121), # (61, 61),
            velocity=7.2,
            angles=[0]
        ))
        self.pivot_tracker = MultiAnglePivotTracker(tracker_config)
        self.pivot_extractor = PivotExtractor()

    def _extract_contours(self, masks):
        '''
            contour: cv2 output contour
            peak: tuple: (x, y)
            basepoint: dict{lbase:(x, y), rbase:(x, y), box:bounding_box}
        '''
        contours = [extract_contour(m) for m in masks]
        contours = [c if (len(c) > 0) else contours[i-1] for (i, c) in enumerate(contours)]
        peaks = [get_peak_by_middle_mfpunet(c) for c in contours]
        basepoints = [adjusted_get_base_points_by_bbox_basepoints(c) for c in contours]
        return contours, peaks, basepoints

    def _extract_pivots_seq(self, contours, peaks, basepoints):
        '''
            pivots_seq.shape == n_frame * n_points * 2
            base points are always at indexes: 0 and -1
            peak point is at len/2
        '''
        assert len(contours) == len(peaks) == len(basepoints), 'something went wrong'
        detected_pivots_sequence = [self.pivot_extractor.detect_pivots(c, p, b, n_segments_per_side=18) \
                                        for c, p, b in zip(contours, peaks, basepoints)]
        pivots_sequence = np.stack(detected_pivots_sequence, axis=0)
        pivots_sequence = smooth_pivots(pivots_sequence, n_iter=3)
        return np.array(pivots_sequence)

    def _preprocess(self, masks):
        contours, peaks, basepoints = self._extract_contours(masks)
        pivots_seq = self._extract_pivots_seq(contours, peaks, basepoints)
        return pivots_seq

    def _estimate_pivots(self, frames, masks, pivots_seq, metadata):
        contours = np.array(pivots_seq)[:, :, None, :]
        initial_pivots = np.array(contours)[0, :, 0] # contours.shape == nFrame * nPoints * 1 * 2
        tracked_pivot_sequence = self.pivot_tracker.track_pivots(
            initial_pivots,
            frames, masks,
            contours, metadata
        )
        #pivot_sequence = smooth_pivots(tracked_pivot_sequence, self.kalman_params, covariance_scale=5)
        return np.array(tracked_pivot_sequence)

    def compute_ef(self, msks, metadata, dataset, output_dir=None):
        def draw(frames, contours, color='green'):
            colors = {'green': (0, 255, 0), 'red': (255, 0, 0)}
            out = []
            for i in range(len(frames)):
                frame, cs = frames[i], contours[i]
                frame = cv2.drawContours(frame, cs, -1, colors[color], 3)
                """
                for ii, pp in enumerate(cs):
                    frame = cv2.putText(frame, f'{ii}', tuple(pp[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                                        1, cv2.LINE_AA)
                iframe = Image.fromarray(frame)
                iframe.save(f"{output_dir}/frame-{i}.jpg")
                """
                out.append(frame)
            return out

        #----------
        frames = [dat['image'] for dat in dataset]
        pivots_seq = self._preprocess(msks)

        multi_pivots_seq = [pivots_seq]
        """
        for d_length in [-7, -14]:
            # extrapolate contour sequences using spline
            extra_seq = [extend_contour(cs, d_length=d_length) for cs in multi_pivots_seq[0]]
            multi_pivots_seq.append(extra_seq)
        """

        smoothed_multi_seq = []
        for seq in multi_pivots_seq:
            # smooth contour sequences using kalman filter
            pivots_seq = self._estimate_pivots(frames.copy(), msks.copy(), seq, metadata) #5
            smoothed_multi_seq.append(pivots_seq)

        if False:
            out_frames = frames.copy()
            seq = smoothed_multi_seq[0]
            out_frames = draw(out_frames, seq[:, :, None], 'green')
            for seq in smoothed_multi_seq[1:]:
                out_frames = draw(out_frames, seq[:, :, None], 'red')
            out_frames = [Image.fromarray(frame) for frame in out_frames]
            out_frames[0].save(f"{output_dir}/vis2.gif", save_all=True, append_images=out_frames, optimize=False, fps=10, loop=0)

        time_window = metadata['window'] # number of frame per heart cycle
        multi_results = [self._compute_stats(pivots_seq, metadata) for pivots_seq in smoothed_multi_seq]
        for (areas, lengths, volumes), pivots_seq in zip(multi_results, multi_pivots_seq):
            #EF, efs, idxEF = compute_EF(time_window, volumes)
            #print(EF, efs, idxEF)
            win_size = int(time_window * 0.9)
            EF, dat = echonet_method(volumes, win_size)

            min_idx, max_idx = dat[2], dat[3]
            GLS, SLS6 = compute_gls(time_window, np.array(pivots_seq), min_idx, max_idx)
            print(GLS)
            print(SLS6)

        #GLS2 = get_gls_by_segments(time_window, pivot_sequence) #11
        #visualize(output_dir, dataset, msks, contours, \
                #pivot_sequence, basepoints, areas, lengths, volumes, self.pivot_tracker)
        #plot(output_dir, areas, lengths, volumes, EF, efs, idxEF, GLS1, GLS2)

        results = {
            "pivot_sequence": np.array(pivots_seq).tolist(),
            "areas": np.array(areas).tolist(),
            "volumes": np.array(volumes).tolist(),
            "min_idx": min_idx,
            "max_idx": max_idx,
            "ef": EF,
            "GLS": GLS,
            "SLS": np.array(SLS6).tolist(),
        }
        return results

        for msk, contour, pivots, area, volume in zip(msks, contours, pivot_sequence, areas, volumes):
            results["contours"].append([ {"x": point[0, 0] / msk.shape[1],
                                          "y": point[0, 1] / msk.shape[0]}
                                    for point in contour])
            results["pivot_sequence"].append([ {"x": point[0] / msk.shape[1],
                                                "y": point[1] / msk.shape[0]}
                                    for point in pivots])
            results["areas"].append(area)
            results["volumes"].append(volume)
        results["ef"] = EF
        results["basepoint_gls"] = GLS1
        results["segmental_gls"] = GLS2
        return results

    def _compute_stats(self, pivots_seq, metadata):
        contours = pivots_seq[:, :, None]
        xscale, yscale = metadata['x_scale'], metadata['y_scale']
        areas = [compute_area(c, xscale, yscale) for c in contours]
        lengths = [compute_lv_length(contour,
                                     contour[len(contour)//2][0],
                                     contour[[0, -1]][:, 0],
                                     xscale, yscale)
                        for contour in contours]
        volumes = np.array([ 8.0 / 3.0 / math.pi * area * area / l[0] for area, l in zip(areas, lengths)])
        return areas, lengths, volumes
