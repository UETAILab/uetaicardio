import logging
from easydict import EasyDict
import numpy as np


logging.basicConfig(level=logging.INFO)


SPEED_DIRECTED_CONFIG = EasyDict(dict(
    # --- DEFAULT OBJECTS ---
    logger=logging.getLogger("default"),

    # --- DATA CONFIG ---
    target_size=(800, 600),
    fps=30,
    heart_rate=75.0,
    window_scale=1.1,
    cell_velocity=7.2,
    
    # --- GEOMETRICAL ANALAYZER ---
    coarse_contour_smooth_filter=np.ones((15, ))/15,
    coarse_contour_threshold=64,
    basepoint_vertical_thresold=0.01,
    basepoint_horizontal_threshold=0.8,
    basepoint_max_step=20,
    n_contour_points_per_side=8,
    fine_contour_smooth_filter=np.ones((21, ))/21,
    fine_contour_n_smooth_iter=3,
    
    # --- KALMAN SMOOTHER ---
    kalman_covariance_scale=30,
    kalman_n_smooth_iter=3,
    kalman_params = EasyDict(dict(
        transition_matrix=[[1, 1, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]],
        observation_matrix=[[1, 0, 0, 0],
                            [0, 0, 1, 0]]
    )),
    
    # --- PIVOT EXTRACTOR ---
    pivots_to_reinitialize=[4, 5],
    pivots_reinitialization_max_distance=61,
    pivots_reinitialization_score_thresold=0.75,
    tracking_kernel_size=(61, 61),
    contour_dilation_rate=1.15,
    contour_contraction_rate=0.85,
    pivots_to_track=[1, 2, 4, 5]#[0, 3, 6]
))

ACCURACY_DIRECTED_CONFIG = EasyDict(dict(
    # --- DEFAULT OBJECTS ---
    logger=logging.getLogger("default"),

    # --- DATA CONFIG ---
    target_size=(800, 600),
    fps=30,
    heart_rate=75.0,
    window_scale=1.1,
    cell_velocity=7.2,
    
    # --- GEOMETRICAL ANALAYZER ---
    coarse_contour_smooth_filter=np.ones((15, ))/15,
    coarse_contour_threshold=64,
    basepoint_vertical_thresold=0.01,
    basepoint_horizontal_threshold=0.8,
    basepoint_max_step=20,
    n_contour_points_per_side=16,
    fine_contour_smooth_filter=np.ones((21, ))/21,
    fine_contour_n_smooth_iter=10,
    
    # --- KALMAN SMOOTHER ---
    kalman_covariance_scale=1,
    kalman_n_smooth_iter=1,
    kalman_params = EasyDict(dict(
        transition_matrix=[[1, 1, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]],
        observation_matrix=[[1, 0, 0, 0],
                            [0, 0, 1, 0]]
    )),
    
    # --- PIVOT EXTRACTOR ---
    pivots_to_reinitialize=[4, 5],
    pivots_reinitialization_max_distance=61,
    pivots_reinitialization_score_thresold=0.75,
    tracking_kernel_size=(61, 61),
    contour_dilation_rate=1.15,
    contour_contraction_rate=0.85,
    pivots_to_track=[1, 2, 4, 5]#[0, 3, 6]
))


DEFAULT = ACCURACY_DIRECTED_CONFIG
