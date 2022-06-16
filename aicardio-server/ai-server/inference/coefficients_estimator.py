import math
import numpy as np
import cv2

from inference.interfaces import InferenceStep
import inference.utils as utils
from inference.config import DEFAULT


class CoefficientsEstimator(InferenceStep):
    def __init__(self, config, logger=DEFAULT.logger):
        r"""Initialize EF-GLS estimator

        Args:
            config (EasyDict-like): Configuration for EF-GLS estimator
                .min_quantile (float): Minimum quantile for computing EF, GLS
                .max_quantile (float): Maximum quantile for computing EF, GLS
        """
        super(CoefficientsEstimator, self).__init__("CoefficientsEstimator",
                                                    logger)
        self.config = config
        self.min_quantile = config.min_quantile
        self.max_quantile = config.max_quantile

    def _validate_inputs(self, item):
        pass

    def _process(self, item):
        metadata = item["metadata"]
        finetuned_contours = item["finetuned_contours"]
        pivot_sequence = item["pivot_sequence"]
        EF, GLS = self.run(metadata, finetuned_contours, pivot_sequence)
        item.update({"EF": EF, "GLS": GLS})
        return item

    def run(self, metadata, finetuned_contours, pivot_sequence):
        r"""Estimate EF and GLS score

        This is the public method used for quick testing purpose

        Args:
            metadata (dict): Metadata extracted from DICOM dataset, which 
                consists of
                    .x_scale (float): Pixel-to-real x-scale
                    .y_scale (float): Pixel-to-real y-scale
                    .window (int): Number of frames per heart cycle.
            finetuned_contours (list(np.array)): Fine-tuned LV contour. This
                is a np.array of shape (n_contour_points, 1, 2), i.e. (x, y)
                coordinates of contour points
            pivot_sequence (np.array): A sequence of sets of LV pivot points.
                This is a np.array of shape (n_frames, n_points, 2), i.e. (x, y)
                coordinates of pivot points across frames
        Returns:
            float: EF score
            float: GLS score
        """
        areas = [self.__get_contour_area(
            c, metadata['x_scale'], metadata['y_scale']
        ) for c in finetuned_contours]
        lengths = [self.__get_contour_length(
            c, pivots[3], pivots[[0, -1]],
            metadata['x_scale'], metadata['y_scale']
        ) for c, pivots in zip(finetuned_contours,
                               pivot_sequence)]
        volumes = np.array([8.0 / 3.0 / math.pi * area * area / l
                            for area, l in zip(areas, lengths)])

        EF, EF_components, idxEF = self.__get_EF(metadata['window'], volumes)
        GLS, GLS_components = self.__get_GLS(metadata['window'], pivot_sequence)
        return EF, EF_components, GLS, GLS_components

    def __get_contour_area(self, contour, x_scale, y_scale):
        scaled_contour = utils.geometry.scale_polygon(
            contour.astype(np.float32),
            x_scale, y_scale
        )
        area = cv2.contourArea(scaled_contour)
        return area

    def __get_contour_length(self, contour, peak, basepoints, x_scale, y_scale):
        middle = basepoints.mean(axis=0)
        ip = utils.geometry.interesection_of_line_and_hull(
            peak, middle,
            np.reshape(contour, (-1, 2)).astype(np.float)
        )
        assert (len(ip) >= 1)
        bottom = max(ip, key=lambda x: x[1]).astype(int)
        contour_length = utils.geometry.scaled_length(
            peak, bottom,
            x_scale, y_scale
        )
        return contour_length

    def __get_EF(self, window, volumes):
        volumes = [(i, v) for i, v in enumerate(volumes) if v is not None]
        if len(volumes) == 0:
            return None, None, None

        efs = []
        for i in range(0, max(1, len(volumes) - window)):
            vs = volumes[i:i + window]
            (idxMin, minV) = self.__get_quantile(vs[window // 3:],
                                                 self.min_quantile)
            (idxMax, maxV) = self.__get_quantile(vs[:window * 2 // 3],
                                                 self.max_quantile)
            ef = (maxV - minV) / maxV * 100
            efs.append((i, ef, idxMin, minV, idxMax, maxV))
        idxEF, EF = self.__get_quantile([(e[0], e[1]) for e in efs], 0.5)
        efs = [e[1] for e in efs]
        return EF, efs, idxEF

    def __get_GLS(self, window, pivot_sequence):
        dist = np.sqrt(np.sum(
            (pivot_sequence[:, 1:, :] - pivot_sequence[:, :-1, :]) ** 2,
            axis=-1)
        )
        GLS_components = [self.__get_EF(window, dist[:, i])[0]
                          for i in range(dist.shape[1])]
        GLS_frames = [self.__get_EF(window, dist[:, i])[1]
                      for i in range(dist.shape[1])]
        GLS_frames = np.mean(GLS_frames, axis=0).tolist()
        GLS = np.mean(GLS_components)
        return GLS, GLS_frames

    def __get_quantile(self, x, quantile):
        '''x (list): each element is of the form (idx, object)'''
        return sorted(x, key=lambda x: x[1])[int(quantile * len(x))]

    def _validate_outputs(self, item):
        if "EF" not in item:
            item["is_valid"] = False
            self.logger.error("No EF score returned")
        if "GLS" not in item:
            item["is_valid"] = False
            self.logger.error("No GLS score returned")

    def _visualize(self, item):
        return item

    def _log(self, item):
        EF, GLS = item["EF"], item["GLS"]
        self.logger.info(f"EF score: {EF} - GLS score {GLS}")
