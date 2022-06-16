import numpy as np


class PivotAnalyzer:
    r"""Pivot analyzer which detects evenly divide the LV contour into multiple segments"""
    def __init__(self):
        pass
    
    def detect_pivots(self, contour, peakpoint, basepoints, n_segments_per_side=3):
        r"""
        
        Args:
            contour (np.array): This is a np.array of shape (n_cnt_points, 1, 2), i.e. (x, y) cooridnates.
            peakpoint (tuple): (x, y) coordinates of peak point
            basepoints (dict): LV base points. This is a Python dict with keys
                .left_basepoint (tuple): (x, y) coordinates of left base point
                .right_basepoint (tuple): (x, y) coordinates of right base point
            n_segments_per_side (int): Desired number of equal contour segments to obtain. 
        Returns:
            np.array: The pivots of the evenly divided contour segments. This is a np.array of shape (n_pivots, 2)
        """
        contour = contour.squeeze(1)
        dist = np.sqrt(np.sum((contour[1:] - contour[:-1])**2, axis=-1))
        peak_pt_idx, left_pt_idx, right_pt_idx, left_diameter, right_diameter = self.__parse_keypoints(contour, dist, peakpoint, basepoints)

        if peak_pt_idx <= left_pt_idx:
            # Get left points
            left_points = [peak_pt_idx]
            for i in range(peak_pt_idx, left_pt_idx):
                if dist[peak_pt_idx:i+1].sum() >= len(left_points) * left_diameter/n_segments_per_side:
                    left_points.append(i+1)
            
            # Get right points
            right_points = [right_pt_idx]
            for i in range(right_pt_idx, len(dist)):
                if dist[right_pt_idx:i+1].sum() >= len(right_points) * right_diameter/n_segments_per_side:
                    right_points.append(i+1)
            for i in range(peak_pt_idx):
                if dist[right_pt_idx:].sum() + dist[:i+1].sum() >= len(right_points) * right_diameter/n_segments_per_side:
                    right_points.append(i+1)
                
        elif peak_pt_idx >= right_pt_idx:
            # Get left points
            left_points = [peak_pt_idx]
            for i in range(peak_pt_idx, len(dist)):
                if dist[i:].sum() >= len(left_points) * left_diameter/n_segments_per_side:
                    left_points.append(i+1)
            for i in range(left_pt_idx):
                if dist[peak_pt_idx:].sum() + dist[:i+1].sum() >= len(left_points) * left_diameter/n_segments_per_side:
                    left_points.append(i+1)
            
            # Get right points
            right_points = [right_pt_idx]
            for i in range(right_pt_idx, peak_pt_idx):
                if dist[right_pt_idx:i+1].sum() >= len(right_points) * right_diameter/n_segments_per_side:
                    right_points.append(i+1)
        
        if len(left_points) < n_segments_per_side+1:
            left_points.append(left_pt_idx)
        if len(right_points) < n_segments_per_side+1:
            right_points.append(peak_pt_idx)
        
        pivots = left_points[::-1] + right_points[::-1][1:]
        pivots = contour[pivots]
        return pivots

    def __parse_keypoints(self, contour, dist, peakpoint, basepoints):
        r"""Parse keypoint data"""
        peak_pt_idx, left_pt_idx, right_pt_idx = self.__get_keypoint_idxs(contour, peakpoint, basepoints)
        left_diameter, right_diameter = self.__get_diameter(dist, peak_pt_idx, left_pt_idx, right_pt_idx)
        return peak_pt_idx, left_pt_idx, right_pt_idx, left_diameter, right_diameter
    
    def __get_keypoint_idxs(self, contour, peakpoint, basepoints):
        r"""Get indices of key points"""
        peak_pt_idx = np.where((contour == peakpoint).all(axis=-1))[0][0]
        left_pt_idx = np.where((contour == basepoints["left_basepoint"]).all(axis=-1))[0][0]
        right_pt_idx = np.where((contour == basepoints["right_basepoint"]).all(axis=-1))[0][0]
        return peak_pt_idx, left_pt_idx, right_pt_idx
    
    def __get_diameter(self, dist, peak_pt_idx, left_pt_idx, right_pt_idx):
        r"""Get LV side diameters"""
        if peak_pt_idx <= left_pt_idx:
            left_diameter = dist[peak_pt_idx:left_pt_idx].sum()
            right_diameter = dist[right_pt_idx:].sum() + dist[:peak_pt_idx].sum()
        elif peak_pt_idx >= right_pt_idx:
            left_diameter = dist[peak_pt_idx:].sum() + dist[:left_pt_idx].sum()
            right_diameter = dist[right_pt_idx:peak_pt_idx].sum()
        return left_diameter, right_diameter