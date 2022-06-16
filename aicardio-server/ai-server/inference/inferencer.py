#!/data.local/giangh/envs/pipeline/bin/python
import os
import cv2
from PIL import Image
from easydict import EasyDict
import argparse

from inference.reader import DICOMReader
from inference.classifiers.classifier import ChamberClassifier
from inference.segmentators.single_frame import (
    SingleFrameDetector,
    SingleFrameSegmentator
)
from inference.geometrical_analyzer.analyzer import (
    CoarseGeometricalAnalyzer,
    FineGeometricalAnalyzer
)
from inference.geometrical_analyzer.pivot_extractor import PivotExtractor
from inference.coefficients_estimator import CoefficientsEstimator

from inference.config import DEFAULT

__all__ = ["Inferencer"]


class Inferencer(object):
    r"""Inference pipeline
    
    The pipeline consists of a sequence of inference steps executed sequentially
    """

    def __init__(self, steps, visualizer=None, verbose=False, debug=False):
        r"""Create an inferencer

        Args:
            steps (list(inference.InferencerStep)): A list of inference steps.
                Inference steps will be executed sequentially.
                * The first step must take an item, i.e. a dict with keys
                    .dicom_path (str): Path to a DICOM file
                * Every step must take an item, i.e. a dict, do something,
                    append the results, i.e. key-value pairs, into the given
                    item
            visualizer (callable): A callable object which takes an item and plot
                something. This is for debug purpose. Default: None
            verbose (bool): Whether or not to print the progress of the pipeline
            debug (bool): Whether or not in DEBUG mode, i.e. visualize results 
                of each inference step. See inference.interfaces.InferenceStep
        """
        self.steps = steps
        self.visualizer = visualizer
        self.verbose = verbose
        self.debug = debug

    def __call__(self, dicom_path):
        r"""Run the inference pipeline on a DICOM file

        Args:
            dicom_path (str): Path to the DICOM file
        Returns:
            dict: A Python dict containing the results of the pipeline, i.e.
                correspond to keys in the dict
        """
        self.item = {"dicom_path": dicom_path}
        for i, step in enumerate(self.steps):
            self.item = step(self.item)
            if self.verbose:
                print(f"Step {i}. {step.name}")
                step.log(self.item)
            if self.debug:
                self.item = step.visualize(self.item)
        return self.item

    def visualize(self):
        if self.visualizer:
            self.item = self.visualizer(self.item)
        return self.item

    def save(self):
        pass


class Visualizer(object):
    def __init__(self, dicom_path, out_dir):
        self.dicom_path = dicom_path
        self.out_dir = out_dir

    def __call__(self, item):
        os.makedirs(self.out_dir, exist_ok=True)
        frames = [Image.fromarray(frame) for frame in item["visualize"]["PivotExtractor"]]
        for i, frame in enumerate(frames):
            save_path = self.__get_save_path(i)
            frame.save(save_path)
        frames[0].save(
            os.path.join(self.out_dir, "vis.gif"),
            save_all=True, append_images=frames[1:],
            duration=30, loop=0
        )

    def __get_save_path(self, idx):
        save_path = os.path.join(self.out_dir, f"{idx}.jpg")
        return save_path


def generate_pipeline_components(args):
    r"""Generate pipeline for inference

    Args:
        args (EasyDict-like): Arguments passed to pipeline.

	    .classifier_module_name (str): Chamber classifier module name
	    .chamber_classifier (str): Model class for chamber classifier
	    .classifier_weights (str): Path to chamber classifier weights

            .detector_model_def (str): PyTorch model config
            .detector_model_weights (str): Path to YOLOv3 .pth weight file
            .detector_batch_size (int): Batch size for LV detector

            .segmentator_module_name (str): Segmentator module name
            .full_frame_segmentator (str): Model class for full frame segmentation
            .full_frame_weights (str): Path to full frame segmentator weights
            .cropped_frame_segmentator (str): Model class for segmentator frame 
            .cropped_frame_weights (str): Path to cropped frame segmentator weights
            
            .device (str): Device to run pipeline
    """
    reader = generate_reader(args)
    classifier = generate_classifier(args)
    detector = generate_detector(args)
    segmentator = generate_segmentator(args)
    coarse_analyzer, fine_analyzer, pivot_extractor = generate_analyzers(args)
    coefficients_estimator = generate_coefficients_estimator(args)
    return (reader, classifier, detector, segmentator,
            coarse_analyzer, fine_analyzer,
            pivot_extractor, coefficients_estimator)


def generate_reader(args):
    reader_config = EasyDict(dict(
        target_size=(800, 600),
        default_fps=DEFAULT.fps,
        default_heart_rate=DEFAULT.heart_rate,
        window_scale=DEFAULT.window_scale,
    ))
    reader = DICOMReader(reader_config)
    return reader


def generate_classifier(args):
    classifier_config = EasyDict(dict(
        module_name=args.classifier_module_name,
        model_class=args.chamber_classifier,
        model_weights=args.classifier_weights,
        image_size=256,
        batch_size=args.classifier_batch_size,
        device=args.device
    ))
    classifier = ChamberClassifier(classifier_config)
    return classifier


def generate_detector(args):
    detector_config = EasyDict(dict(
        model_def=args.detector_model_def,
        image_size=416,
        model_weights=args.detector_model_weights,
        batch_size=args.detector_batch_size,
        device=args.device
    ))
    detector = SingleFrameDetector(detector_config)
    return detector


def generate_segmentator(args):
    segmentator_config = EasyDict(dict(
        full_image_config=EasyDict(dict(
            module_name=args.segmentator_module_name,
            model_class=args.full_frame_segmentator,
            model_weights=args.full_frame_weights,
        )),
        cropped_image_config=EasyDict(dict(
            module_name=args.segmentator_module_name,
            model_class=args.cropped_frame_segmentator,
            model_weights=args.cropped_frame_weights,
        )),
        device=args.device,
        image_size=256
    ))
    segmentator = SingleFrameSegmentator(segmentator_config)
    return segmentator


def generate_analyzers(args):
    coarse_analyzer = CoarseGeometricalAnalyzer()
    fine_analyzer = FineGeometricalAnalyzer()

    pivot_extractor_config = EasyDict(dict(
        reinitialized_pivots=DEFAULT.pivots_to_reinitialize,
        tracked_pivots=DEFAULT.pivots_to_track
    ))
    pivot_extractor = PivotExtractor(pivot_extractor_config)
    return coarse_analyzer, fine_analyzer, pivot_extractor


def generate_coefficients_estimator(args):
    coefficients_estimator_config = EasyDict(dict(
        min_quantile=0.05,
        max_quantile=0.95
    ))
    coefficients_estimator = CoefficientsEstimator(coefficients_estimator_config)
    return coefficients_estimator
