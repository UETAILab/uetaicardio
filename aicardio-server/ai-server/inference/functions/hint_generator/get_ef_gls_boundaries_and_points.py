import os
import glob
import numpy as np
from scipy import signal
import cv2


__all__ = ["get_ef_gls_boundaries_and_points"]


def get_ef_gls_boundaries_and_points(frame, boundary, pivots, gls_extend_size=0.05):
    r"""Reorder boundary to clockwise order and remove boundary points along the LV baseline to produce EF boundaries and points
    Extend EF boundaries and points to get GLS boundaries and points
    
    Args:
        boundary (np.array): (x, y) coordinates, i.e. np.array of shape (n_points, 1, 2)
        pivots (np.array): (x, y) coordinates, i.e. np.array oif shape (n_points, 2)
    Returns:
        np.array: (x, y) coordinates, i.e. np.array of shape (n_points, 1, 2)
        list(int): Indices of new pivot points
    """
    ef_pivots = pivots
    ef_boundary, ef_pivots_idx = reorder_boundary_and_pivots(boundary[:, 0, :], ef_pivots)
    gls_boundary = get_gls_boundary(frame.shape, ef_boundary, gls_extend_size)
    pos_axis = (ef_pivots[0] + ef_pivots[-1])/2 - ef_pivots[3]
    gls_boundary = get_clockwise_contour(gls_boundary, pos_axis)
    gls_boundary = clip_gls_contour(ef_pivots, gls_boundary)
    gls_pivots_idx = get_gls_pivots_idx(gls_boundary, ef_boundary, ef_pivots_idx)

    ef_boundary = ef_boundary[:, None, :]
    ef_pivots = ef_boundary[ef_pivots_idx, 0]
    gls_boundary = gls_boundary[:, None, :]
    gls_pivots = gls_boundary[gls_pivots_idx, 0]
    return ef_pivots, ef_boundary, gls_pivots, gls_boundary

def reorder_boundary_and_pivots(boundary, pivots):
    pivots_idx = get_pivots_idx(pivots, boundary)
    new_boundary = np.concatenate([boundary[:pivots_idx[0]+1][::-1], boundary[pivots_idx[-1]:][::-1]], axis=0).astype(int)
    new_pivots_idx = get_pivots_idx(pivots, new_boundary)
    return new_boundary, new_pivots_idx

def get_pivots_idx(pivots, contour):
    pivots_idx = [get_nearest_contour_point(pivot, contour) for pivot in pivots]
    return pivots_idx

def get_nearest_contour_point(point, contour):
    dist = np.sum((point - contour)**2, axis=-1)
    return np.argmin(dist)

def get_clockwise_contour(contour, pos_axis):
    center = np.mean(contour, axis=0)
    normalized_contour = contour - center
    angles = np.arctan2(normalized_contour[:, 1], normalized_contour[:, 0]) - np.arctan(pos_axis[1:], pos_axis[:1])
    angles[angles < 0] += 2*np.pi
    clockwise_order = np.argsort(angles)
    contour = contour[clockwise_order]
    return contour

def get_gls_boundary(frame_shape, boundary, extend_size):
    dist = get_distance_map(frame_shape, boundary)
    ret, mask = cv2.threshold(dist, extend_size, 255, 0)
    mask = np.uint8(255 - mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max([contour for contour in contours],         
                          key=lambda x:cv2.contourArea(x))
    return largest_contour[:, 0, :]

def get_distance_map(image_shape, boundary):
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask = cv2.fillPoly(mask, boundary[None, :, :], (255, 255, 255))
    mask = cv2.cvtColor(255 - mask, cv2.COLOR_BGR2GRAY)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    return dist

def clip_gls_contour(ef_pivots, gls_boundary):
    normal_vec = np.array([
        ef_pivots[0, 1] - ef_pivots[-1, 1],
        ef_pivots[-1, 0] - ef_pivots[0, 0]
    ])
    dot = np.sum(gls_boundary * normal_vec, axis=-1) - np.sum(ef_pivots[0] * normal_vec, axis=-1)
    new_gls_boundary = gls_boundary[dot <= 0]
    return new_gls_boundary

def get_gls_pivots_idx(gls_boundary, ef_boundary, ef_pivots_idx):
    ef_dist = np.sqrt(np.sum((ef_boundary[1:] - ef_boundary[:-1])**2, axis=-1))
    ef_contour_len = ef_dist.sum()
    ef_segment_relative_len = [ef_dist[ef_pivots_idx[i]:ef_pivots_idx[i+1]].sum()/ef_contour_len
                               for i in range(len(ef_pivots_idx)-1)]
    ef_segment_relative_len = np.cumsum(ef_segment_relative_len)
    
    gls_dist = np.sqrt(np.sum((gls_boundary[1:] - gls_boundary[:-1])**2, axis=-1))
    gls_dist = np.insert(gls_dist, 0, 0)
    gls_dist = np.cumsum(gls_dist) / gls_dist.sum()
    gls_pivots_idx = [0] + [np.argmax(gls_dist >= l) for l in ef_segment_relative_len]
    return gls_pivots_idx
