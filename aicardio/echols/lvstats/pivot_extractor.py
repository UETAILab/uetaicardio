import numpy as np


class PivotExtractor:
    r"""Pivot analyzer which detects evenly divide the LV contour into multiple segments"""

    def _get_keypoint_idxs(self, contour, peakpoint, basepoints):
        peak_idx = np.where((contour == peakpoint).all(axis=-1))[0][0]
        lbase_idx = np.where((contour == basepoints["lbase"]).all(axis=-1))[0][0]
        rbase_idx = np.where((contour == basepoints["rbase"]).all(axis=-1))[0][0]
        return peak_idx, lbase_idx, rbase_idx

    def _get_side_diameter(self, dist, peak_idx, lbase_idx, rbase_idx):
        if peak_idx <= lbase_idx:
            l_diameter = dist[peak_idx:lbase_idx].sum()
            r_diameter = dist[rbase_idx:].sum() + dist[:peak_idx].sum()
        elif peak_idx >= rbase_idx:
            l_diameter = dist[peak_idx:].sum() + dist[:lbase_idx].sum()
            r_diameter = dist[rbase_idx:peak_idx].sum()
        return l_diameter, r_diameter

    def _parse_keypoints(self, contour, dist, peakpoint, basepoints):
        peak_idx, lbase_idx, rbase_idx = self._get_keypoint_idxs(contour, peakpoint, basepoints)
        l_diameter, r_diameter = self._get_side_diameter(dist, peak_idx, lbase_idx, rbase_idx)
        return peak_idx, lbase_idx, rbase_idx, l_diameter, r_diameter

    def detect_pivots(self, contour, peakpoint, basepoints, n_segments_per_side=17):
        r"""
            n_segments_per_side (int): Desired number of equal contour segments to obtain.
        Returns:
            The pivots of the evenly divided contour segments. This is a np.array of shape (n_pivots, 2)
        """
        contour = contour.squeeze(1)
        dist = np.sqrt(np.sum((contour[1:] - contour[:-1])**2, axis=-1))
        peak_pt_idx, left_pt_idx, right_pt_idx, left_diameter, right_diameter = self._parse_keypoints(contour, dist, peakpoint, basepoints)
        peak_idx, lbase_idx, rbase_idx, l_diameter, r_diameter = self._parse_keypoints(contour, dist, peakpoint, basepoints)

        if peak_idx <= lbase_idx:
            lside_ids = [peak_idx]
            for i in range(peak_idx, lbase_idx):
                if dist[peak_idx:i+1].sum() >= (len(lside_ids) * l_diameter/n_segments_per_side):
                    lside_ids.append(i+1)

            rside_ids = [rbase_idx]
            for i in range(rbase_idx, len(dist)):
                if dist[rbase_idx:i+1].sum() >= (len(rside_ids) * r_diameter/n_segments_per_side):
                    rside_ids.append(i+1)
            for i in range(peak_idx):
                if dist[rbase_idx:].sum() + dist[:i+1].sum() >= len(rside_ids) * r_diameter/n_segments_per_side:
                    rside_ids.append(i+1)
        elif peak_idx >= rbase_idx:
            lside_ids = [peak_idx]
            for i in range(peak_idx, len(dist)):
                if dist[i:].sum() >= len(lside_ids) * l_diameter/n_segments_per_side:
                    lside_ids.append(i+1)
            for i in range(lbase_idx):
                if dist[peak_idx:].sum() + dist[:i+1].sum() >= len(lside_ids) * l_diameter/n_segments_per_side:
                    lside_ids.append(i+1)

            rside_ids = [rbase_idx]
            for i in range(rbase_idx, peak_idx):
                if dist[rbase_idx:i+1].sum() >= len(rside_ids) * r_diameter/n_segments_per_side:
                    rside_ids.append(i+1)

        if len(lside_ids) < n_segments_per_side+1:
            lside_ids.append(lbase_idx)
        if len(rside_ids) < n_segments_per_side+1:
            rside_ids.append(peak_idx)

        pivots = lside_ids[::-1] + rside_ids[::-1][1:]
        #pivots = lside_ids + rside_ids[:-1]
        pivots = contour[pivots]
        return pivots
