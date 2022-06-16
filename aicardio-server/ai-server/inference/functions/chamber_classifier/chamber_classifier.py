#!/data.local/giangh/envs/pipeline/bin/python
import os
import glob
import json
import pandas as pd
import cv2
import pydicom as dcm
from easydict import EasyDict
import argparse

from inference.inferencer import  Inferencer, generate_pipeline_components


CHAMBER2IDX = {"2C": 0, "3C": 1, "4C": 2, "none": 3}


def load_pipeline_config(config_path):
    r"""Load pipeline config

    Args:
        config_path (str): Path to .json config file, with fields
            .classifier_module_name (str): Chamber classifier module name
            .chamber_classifier (str): Model class for chamber classifier
            .classifier_weights (str): Path to chamber classifier weights
            .device (str): Device to run pipeline
    Returns:
        EasyDict: Config for pipeline
    """
    with open(config_path) as f:
        config = json.load(f)
    return EasyDict(config)


def load_pipeline(pipeline_config):
    (reader, classifier, detector, segmentator, 
     coarse_analyzer, fine_analyzer,
     pivot_extractor, coefficients_estimator) = generate_pipeline_components(pipeline_config) 
    inferencer = Inferencer([
        reader,
	classifier,
    ])
    return inferencer

def process_dicom_dir(args, root):
    r"""Process a DICOM directory. The results will be recorded within a
    JSON file named as "results.json" located within the directory
        `os.path.join(args.out_dir, os.path.basename(root))`

    Args:
        args (EasyDict-like): Contain configuration for the running
        root (str): Path to directory. The directory must be of the form
            |-- root
                |-- case 1
                    |-- dicom file 1
                    |-- dicom file 2
                |-- case 2
                    |-- ...
                |-- ...
    """
    pipeline_config = load_pipeline_config(args.pipeline_config)
    inferencer = load_pipeline(pipeline_config)

    results = {}
    paths = get_dicom_paths(root)
    for i, path in enumerate(paths):
        print(f"PROGRESS {i}/{len(paths)} - File name: {path}")
        try:
            item = inferencer(path)
            results[path] = item["chamber"] 
        except Exception as e:
            print(f"\t{e}")
        if i%100 == 0:
            save_results(args.out_dir, root, results)
    save_results(args.out_dir, root, results)

def get_dicom_paths(root):
    paths = glob.glob(os.path.join(root, "*", "*"))
    return paths

def save_results(out_dir, root, results):
    out_dir = os.path.join(out_dir, os.path.basename(root))
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"results.json")
    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config", type=str, help="Pipeline config",
                        default="/data.local/giangh/pipeline/inference/functions/hint_generator_pipeline_config/2C_example.json")
    parser.add_argument("--out_dir", type=str, help="Output dir",
                        default="tmp/hints")
    args = parser.parse_args()

    dirs = glob.glob(os.path.join("/data.local/data/DICOM_DATA/", "*"))
    dirs = [_dir for _dir in dirs if "OTHER_DATA" not in _dir]
    [process_dicom_dir(args, _dir) for _dir in dirs]
