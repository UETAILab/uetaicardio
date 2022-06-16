#!/data.local/giangh/envs/pipeline/bin/python
import os
import glob
import time
import threading
import json
import pandas as pd
import cv2
import pydicom as dcm
from easydict import EasyDict
from queue import PriorityQueue
import argparse

from inference.inferencer import  (
    Inferencer, generate_reader, generate_classifier
)
from inference.interfaces import ThreadedStep


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


def generate_threaded_pipeline_components(
        pipeline_config, out_dir, root, 
        n_readers=10, queue_maxsize=2000
    ):
    input_queue = get_input_queue(root)
    reader_output_queue = PriorityQueue(maxsize=queue_maxsize)
    readers = [
        ThreadedStep(
            generate_reader(pipeline_config), 
            input_queue, reader_output_queue,
            n_end_items=1
        ) for i in range(n_readers)
    ]

    classifier_output_queue = PriorityQueue(maxsize=queue_maxsize)
    classifier = ThreadedStep(generate_classifier(pipeline_config), 
                              reader_output_queue, 
                              classifier_output_queue,
                              n_end_items=n_readers)

    saver = threading.Thread(
        target=save_results, 
        args=(out_dir, root, classifier_output_queue, n_readers)
    )
    return dataflow, readers, classifier, saver, input_queue, classifier_output_queue


def get_input_queue(root): 
    input_queue = PriorityQueue()
    paths = get_dicom_paths(root)
    for i, path in enumerate(paths):
        out_queue.put((1, i, {"dicom_path": path}))
        time.sleep(1e-5)

    for i in range(args.n_readers):
        out_queue.put((2, i, None))
        time.sleep(1e-5)

def get_dicom_paths(root):
    paths = glob.glob(os.path.join(root, "*", "*"))
    return paths


def save_results(out_dir, root, in_queue, n_end_items):
    n_collected_end_items = 0
    n_finished = 0
    n_saved = 0
    results = {}
    while True:
        if not in_queue.empty():
            priority, idx, item = in_queue.get()
            if priority == 1:
                n_finished += 1
                print(f"Finished {n_finished} file - path: "+item["dicom_path"])
                results[item["dicom_path"]] = item["chamber"]
                if len(results) % 100 == 0:
                    save_to_json(out_dir, root, results, n_saved)
                    n_saved += 1
                    results = {}
            else:
                n_collected_end_items += 1

        if n_collected_end_items == n_end_items:
            print("Terminate thread Saver")
            save_to_json(out_dir, root, results, n_saved)
            break

        time.sleep(1e-5)


def save_to_json(out_dir, root, results, idx):
    out_dir = os.path.join(out_dir, os.path.basename(root))
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"results-{idx}.json")
    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {output_file}")


def process_dicom_dir(args):
    pipeline_config = load_pipeline_config(args.pipeline_config)
    (dataflow, readers, classifier, saver, 
     in_queue, out_queue) = generate_threaded_pipeline_components(
        pipeline_config, args.out_dir, args.root,
        args.n_readers,  args.queue_maxsize
    )
    threads = [dataflow] + readers + [classifier] + [saver]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Root to DICOM dir")
    parser.add_argument("--n_readers", type=int, help="Number of threads for reading DICOM",
                        default=10)
    parser.add_argument("--queue_maxsize", type=int, help="Maximum number of items to read at one time",
                        default=2000)
    parser.add_argument("--pipeline_config", type=str, help="Pipeline config",
                        default="/data.local/giangh/pipeline/inference/functions/pipeline_config/2C_example.json")
    parser.add_argument("--out_dir", type=str, help="Output dir",
                        default="tmp/hints")
    args = parser.parse_args()

    process_dicom_dir(args)
