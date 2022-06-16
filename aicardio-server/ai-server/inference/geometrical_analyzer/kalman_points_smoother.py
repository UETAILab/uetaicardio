import time
import numpy as np
from pykalman import KalmanFilter
from inference.config import DEFAULT


class KalmanPointsSmoother(object):
    def __init__(self,
                 covariance_scale=DEFAULT.kalman_covariance_scale,
                 n_smooth_iter=DEFAULT.kalman_n_smooth_iter,
                 kalman_params=DEFAULT.kalman_params):
        self.covariance_scale = covariance_scale
        self.n_smooth_iter = n_smooth_iter
        self.kalman_params = kalman_params

    def __call__(self, point_sequence):
        point_sequence = np.concatenate([points[None, ...] 
                                         for points in point_sequence])
        point_sequence = self.__smooth_points(point_sequence)
        return point_sequence
        
    def __smooth_points(self, point_sequence):
        for i in range(point_sequence.shape[1]):
            initial_state_mean = [point_sequence[0, i, 0], 0, 
                                  point_sequence[0, i, 1], 0]
            observation_covariance = self.__estimate_observation_covariance(
                i, initial_state_mean,
                point_sequence
            )
            point_sequence = self.__kalman_smooth(
                i, point_sequence, 
                initial_state_mean, 
                observation_covariance
            )
        return point_sequence
    
    def __estimate_observation_covariance(self, point_idx,
                                          initial_state_mean, point_sequence):
        kf = KalmanFilter(
            transition_matrices=self.kalman_params.transition_matrix, 
            observation_matrices=self.kalman_params.observation_matrix, 
            initial_state_mean=initial_state_mean
        )
        kf = kf.em(point_sequence[:, point_idx], n_iter=self.n_smooth_iter)
        observation_covariance = self.covariance_scale*kf.observation_covariance
        return observation_covariance
    
    def __kalman_smooth(self, point_idx, point_sequence, 
                        initial_state_mean, 
                        observation_covariance):
        kf = KalmanFilter(
            transition_matrices=self.kalman_params.transition_matrix,
            observation_matrices=self.kalman_params.observation_matrix,
            initial_state_mean=initial_state_mean,
            observation_covariance=observation_covariance,
            em_vars=['transition_covariance', 'initial_state_covariance']
        )
        kf = kf.em(point_sequence[:, point_idx], n_iter=self.n_smooth_iter)
        smoothed_state_means, smoothed_state_covariances = kf.smooth(
            point_sequence[:, point_idx]
        )
        
        point_sequence[:, point_idx, 0] = smoothed_state_means[:, 0]
        point_sequence[:, point_idx, 1] = smoothed_state_means[:, 2]
        return point_sequence
