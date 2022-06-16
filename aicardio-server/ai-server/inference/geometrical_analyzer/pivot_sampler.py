import numpy as np


class PivotSampler(object):
    r"""Pivot analyzer which detects evenly divide the LV contour into multiple 
    segments"""
    def __init__(self):
        pass
    
    def __call__(self, contour, keypoints, n_segments_per_side=3):
        r"""
        
        Args:
            contour (np.array): This is a np.array of shape (n_cnt_points, 1, 2)
                i.e. (x, y) cooridnates.
            keypoints (tuple): Peak point and two base points
                peakpoint (np.array): (x, y)-coordinates of peak point
                basepoints (tuple): LV base points.
                    .np.array: (x, y)-coordinates of left base point
                    .np.array: (x, y)-coordinates of right base point
            n_segments_per_side (int): Desired number of equal contour segments 
                to obtain. 
        Returns:
            np.array: The pivots of the evenly divided contour segments. This is 
                a np.array of shape (n_pivots, 2)
        """
        contour = contour.squeeze(1)
        peak, basepoints = keypoints

        dist = np.sqrt(np.sum((contour[1:] - contour[:-1])**2, axis=-1))
        (peak_idx, left_idx, right_idx, 
         left_diameter, right_diameter) = self.__parse_keypoints(contour, dist, 
                                                                 peak, basepoints)
        
        left_points = self.__get_left_pivots(peak_idx, left_idx, left_diameter,
                                             dist, n_segments_per_side,
                                             peak_idx <= left_idx)
        right_points = self.__get_right_pivots(peak_idx, right_idx, right_diameter,
                                               dist, n_segments_per_side,
                                               peak_idx <= left_idx)
        pivots = left_points[::-1] + right_points[::-1][1:]
        pivots = contour[pivots]
        return pivots

    def __parse_keypoints(self, contour, dist, peakpoint, basepoints):
        r"""Parse keypoint data"""
        peak_idx, left_idx, right_idx = self.__get_keypoint_idxs(contour, peakpoint, basepoints)
        left_diameter, right_diameter = self.__get_diameter(dist, peak_idx, left_idx, right_idx)
        return peak_idx, left_idx, right_idx, left_diameter, right_diameter
    
    def __get_keypoint_idxs(self, contour, peakpoint, basepoints):
        r"""Get indices of key points"""
        peak_idx = np.where((contour == peakpoint).all(axis=-1))[0][0]
        left_idx = np.where((contour == basepoints[0]).all(axis=-1))[0][0]
        right_idx = np.where((contour == basepoints[1]).all(axis=-1))[0][0]
        return peak_idx, left_idx, right_idx
    
    def __get_diameter(self, dist, peak_idx, left_idx, right_idx):
        r"""Get LV side diameters"""
        if peak_idx <= left_idx:
            left_diameter = dist[peak_idx:left_idx].sum()
            right_diameter = dist[right_idx:].sum() + dist[:peak_idx].sum()
        elif peak_idx >= right_idx:
            left_diameter = dist[peak_idx:].sum() + dist[:left_idx].sum()
            right_diameter = dist[right_idx:peak_idx].sum()
        return left_diameter, right_diameter
    
    def __get_left_pivots(self, peak_idx, left_idx, left_diameter, 
                          dist, n_segments_per_side, peak_first=True):
        if peak_first:
            left_points = [peak_idx]
            for i in range(peak_idx, left_idx):
                if dist[peak_idx:i+1].sum() >= len(left_points) * left_diameter/n_segments_per_side:
                    left_points.append(i+1)
        else:
            left_points = [peak_idx]
            for i in range(peak_idx, len(dist)):
                if dist[i:].sum() >= len(left_points) * left_diameter/n_segments_per_side:
                    left_points.append(i+1)
            for i in range(left_idx):
                if dist[peak_idx:].sum() + dist[:i+1].sum() >= len(left_points) * left_diameter/n_segments_per_side:
                    left_points.append(i+1)
        if len(left_points) < n_segments_per_side+1:
            left_points.append(left_idx)
        return left_points
    
    def __get_right_pivots(self, peak_idx, right_idx, right_diameter,
                           dist, n_segments_per_side, peak_first=True):
        if peak_first:
            right_points = [right_idx]
            for i in range(right_idx, len(dist)):
                if dist[right_idx:i+1].sum() >= len(right_points) * right_diameter/n_segments_per_side:
                    right_points.append(i+1)
            for i in range(peak_idx):
                if dist[right_idx:].sum() + dist[:i+1].sum() >= len(right_points) * right_diameter/n_segments_per_side:
                    right_points.append(i+1)
        else:
            right_points = [right_idx]
            for i in range(right_idx, peak_idx):
                if dist[right_idx:i+1].sum() >= len(right_points) * right_diameter/n_segments_per_side:
                    right_points.append(i+1)
        if len(right_points) < n_segments_per_side+1:
            right_points.append(peak_idx)
        return right_points