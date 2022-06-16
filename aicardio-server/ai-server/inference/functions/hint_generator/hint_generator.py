#!/data.local/giangh/envs/pipeline/bin/python
import os
import glob
import time
import json
import cv2
import pydicom as dcm
from easydict import EasyDict
import multiprocessing
from itertools import repeat
from queue import Queue
import argparse

from inference.inferencer import  (
    Inferencer, generate_reader, generate_classifier,
    generate_detector, generate_segmentator,
    generate_analyzers, generate_coefficients_estimator,
    generate_pipeline_components
)
from inference.functions.hint_generator.get_ef_gls_boundaries_and_points import get_ef_gls_boundaries_and_points 


CHAMBER2IDX = {"2C": 0, "3C": 1, "4C": 2, "none": 3}
IDX2CHAMBER = {0: "2C", 1: "3C", 2: "4C", 3: "none"}


def load_pipeline_config(config_path):
    r"""Load pipeline config

    Args:
        dicom_path (str): Path to DICOM file
        config_path (str): Path to .json config file, with fields
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
    Returns:
        EasyDict: Config for pipeline
    """
    with open(config_path) as f:
        config = json.load(f)
    return EasyDict(config)


def run(json_path, chamber, pipeline_config, out_dir,
        gls_extend_size, n_threads=10):
    r"""
    
    Args:
        json_path (str): Path to JSON file of the form
            {
                <path 1>: <chamber 1>,
                <path 2>: <chamber 2>,
                ...
            }
        pipeline_config (EasyDict): Configuration for inference pipeline
        out_dir (str): Path to output directory
	gls_extend_size (float): Size between EF boundary and GLS boundary
        n_threads (int): Number of processes to process DICOM data
    """
    chamber_data = load_chamber_data(json_path)
    print("Finished loading chamber data")
    input_queue = get_input_queue(chamber_data, chamber)
    print("Loaded input queue")
    
    processes = [
        multiprocessing.Process(
            target=process_dicom,
            args=(pipeline_config, input_queue, out_dir, gls_extend_size)
        ) for i in range(n_threads)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

def load_chamber_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def get_input_queue(chamber_data, chamber):
    input_queue = multiprocessing.Manager().Queue()
    for i, dicom_path in enumerate(chamber_data):
        if chamber_data[dicom_path] == chamber:
            input_queue.put(dicom_path)
    return input_queue

def process_dicom(pipeline_config, input_queue, out_dir,
		  gls_extend_size):
    print("Initialized a process")
    inferencer = load_inferencer(pipeline_config)
    while not input_queue.empty():
        dicom_path = input_queue.get()
        try:
            item = inferencer(dicom_path)
            save_result(item, out_dir, gls_extend_size)
            print(f"Finished file {dicom_path}")
        except Exception as e:
            print(f"Error at {dicom_path} - {e}")
        time.sleep(1e-5)

def load_inferencer(pipeline_config):
    (reader, classifier, detector, segmentator,
     coarse_analyzer, fine_analyzer,
     pivot_extractor, coefficients_estimator) = generate_pipeline_components(pipeline_config)
    inferencer = Inferencer([
        reader, classifier, detector, segmentator,
        coarse_analyzer, fine_analyzer, pivot_extractor,
        coefficients_estimator
    ])
    return inferencer

def save_result(item, out_dir, gls_extend_size):
    result = {
        "dicomDiagnosis": {
            "chamber": item["chamber"],
            "chamber_idx": CHAMBER2IDX[item["chamber"]],
            "not_standard": False,
            "lad": False,
            "rca": False,
            "lcx": False,
            "points": [],
            "note": ""
        },
        "dicomAnnotation": get_framewise_annotation(item, gls_extend_size)
    }
    to_json(item, result, out_dir)

def get_framewise_annotation(item, gls_extend_size):
    dicom_path = item["dicom_path"]
    contours = item["finetuned_contours"]
    pivot_sequence = item["pivot_sequence"]
    
    _, h, w, _ = item["frames"].shape
    annotations = []
    for i, (frame, contour, pivots) in enumerate(zip(
	item["frames"],
    	item["finetuned_contours"],
	item["pivot_sequence"]
    )):
        (ef_point, ef_boundary, 
	 gls_point, gls_boundary) = get_ef_gls_boundaries_and_points(
	    frame, contour, pivots, gls_extend_size
	)
        annotation = {
            "ef_point": [
	        {"x": point[0]/w, "y": point[1]/h}
		for point in ef_point
	    ],
            "ef_boundary": [[
	        {"x": point[0, 0]/w, "y": point[0, 1]/h}
		for point in ef_boundary
	    ]],
            "gls_point": [
	        {"x": point[0]/w, "y": point[1]/h}
		for point in gls_point
	    ],
            "gls_boundary": [[
	        {"x": point[0, 0]/w, "y": point[0, 1]/h}
		for point in gls_boundary
	    ]],
            "is_ESV": False, "is_EDV": False,
            "measure_length": [],
            "measure_area": [[]],
            "length": 0, "area": 0, "volume": 0,
            "ef_manual": 0, "indexFrame": i,
            "indexESV": -1, "indexEDV": -1,
            "volumeESV": 0, "volumeEDV": 0,
            "has_EF_draw": True, "has_GLS_draw": True
        }
        annotations.append(annotation)
    return annotations

def to_json(item, result, out_dir):
    out_path = get_out_path(item, out_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)

def get_out_path(item, out_dir):
    dicom_path = item["dicom_path"]
    _dir, dicom_id = os.path.split(dicom_path)
    case_id = os.path.basename(_dir)
    out_path = os.path.join(out_dir, case_id, f"{dicom_id}.json")
    return out_path


if __name__ == "__main__":
    r"""
    .root (str) Directory of the form
        |-- root
            |-- BVE
            |-- BVHNVX
            |-- ...
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to directory containing JSON chamber file")
    parser.add_argument("--chamber", type=str, help="Chamber to run")
    parser.add_argument("--out_dir", type=str, help="Path to output directory")
    parser.add_argument("--pipeline_config", type=str, help="Pipeline config",
                        default="/data.local/giangh/pipeline/inference/functions/pipeline_config/2C_example.json")
    parser.add_argument("--gls_extend_size", type=float, help="Must be within 0..1, i.e. distance from EF boundary to GLS boundary",
                        default=0.05)
    parser.add_argument("--n_threads", type=int, default=2)
    args = parser.parse_args()

    pipeline_config = load_pipeline_config(args.pipeline_config)
    
    json_paths = glob.glob(os.path.join(args.root, "*", "results.json"))
    for json_path in json_paths:
        hospital = os.path.basename(os.path.dirname(json_path))
        out_dir = os.path.join(args.out_dir, hospital)
        run(json_path, args.chamber, pipeline_config, out_dir,
            args.gls_extend_size, args.n_threads)
