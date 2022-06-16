import os
import time
import json
from easydict import EasyDict
import datetime
import argparse

import pandas as pd


from lv_inference.datasets import DICOMDataset
from lv_inference.doubleframe_dataset import DoubleFrameDICOMDataset
from lv_inference.dicom_segmentator import DICOMSegmentator
from lv_inference.ef_computer import EFComputer


class DICOMInferencer:
    def __init__(self, args):
        self.config = args
        self.segmentator = DICOMSegmentator(args)
        self.ef_computer = EFComputer()

    def compute_ef(self, dataset, output_dir=None):
        masks = self.segmentator.get_segmentation_masks(dataset)
        results = self.ef_computer.compute_ef(
            masks, dataset.metadata, 
            dataset, output_dir
        )
        return results


if __name__ == "__main__":
    r""" 
    Detetor: ../22022020-wrapped_pipeline/weights/lv_detection_model.pth
    Full frame segmentator: ../22022020-wrapped_pipeline/weights/full_image_segmentation_model.pth
    Cropped frame segmentator: ../22022020-wrapped_pipeline/weights/cropped_image_segmentation_model.pth
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_path", type=str, help="DICOM path")
    parser.add_argument("--segmentation_image_size", type=int, default=256, help="input tensor size")
    parser.add_argument("--visualize_dir", type=str, default=None, help="visualization directory (not None then visualize outputs)")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--device", type=str, default="cuda:0", help="number of cpu threads to use during batch generation")

    parser.add_argument("--detector_model_def", type=str, default="PyTorch_YOLOv3/config/yolov3-custom.cfg", help="detector .cfg file")
    parser.add_argument("--detector_image_size", type=int, default=416, help="detector")
    parser.add_argument("--detector_model_weights", type=str, 
                        default="/data.local/data/models/YOLO-2C/0.7435_yolov3_ckpt_75.pth",
                        help="detector .pth weights file")
    parser.add_argument("--detector_batch_size", type=int, default=2,
                        help="detector batch size")

    parser.add_argument("--full_image_module_name", type=str, default="segmentation", help="module / package name")
    parser.add_argument("--full_image_segmentation_class", type=str, default="AuxUNet", 
                        help="full-image segmentator class name, eg., MobileNetv2_DeepLabv3")
    parser.add_argument("--full_image_segmentation_weights", type=str, 
                        default="/data.local/data/models/segmentation/full2C/full_aux_giangunet_invhd_0009_0.8604_best.pth",
                        help="trained full-image segmentator weights path")
    
    parser.add_argument("--cropped_image_module_name", type=str, default="segmentation", help="module / package name")
    parser.add_argument("--cropped_image_segmentation_class", type=str, default="AuxUNet",
                        help="cropped-image segmentator class name, eg., MobileNetv2_DeepLabv3")
    parser.add_argument("--cropped_image_segmentation_weights", type=str, 
                        default="/data.local/data/models/segmentation/cropped2C/cropped_aux_bce_invhd_giangunet_0014_0.8642_best.pth", 
                        help="trained cropped-image segmentator weights path")
    parser.add_argument("--csv_output_file", type=str,
                        default="tmp/ef_gls_results.csv",
                        help="csv file to write data")
    parser.add_argument("--frame_start_json_file", type=str,
                        default=None, help="json file containing frame start data")
    parser.add_argument("--chamber", type=str,
                        default=None, help="chamber type")
    args = parser.parse_args()
    
    case_idx = args.dicom_path[args.dicom_path.find("__")-4:args.dicom_path.find("__")]
    case_idx = str(case_idx)
    case_name = args.dicom_path[args.dicom_path.find("__")+2:args.dicom_path.rfind("/")]
    if args.frame_start_json_file is not None:
        with open(args.frame_start_json_file) as f:
            frame_start_EF_GLS = json.load(f)
        if case_idx in frame_start_EF_GLS:
            frame_start_EF_GLS = frame_start_EF_GLS[case_idx][args.chamber]
        else:
            frame_start_EF_GLS = 0
    else:
        frame_start_EF_GLS = 0
    print(f"Frame start: {frame_start_EF_GLS}")

    print(f"Processing file {args.dicom_path}")
    data_config  = EasyDict(dict(case_idx=case_idx,
                                 dicom_path=args.dicom_path,
                                 target_size=(800, 600),
                                 default_heart_rate=75.0,
                                 default_fps=30,
                                 frame_start=frame_start_EF_GLS))
    dataset = DICOMDataset(data_config)
    print("Loaded dataset successfully!")
    
    inferer = DICOMInferencer(args)
    print("Compute ef ...")
    start = time.time()
    results = inferer.compute_ef(dataset, args.visualize_dir)
    runtime = time.time()-start
    print(f"Total runtime: {runtime:.4f} - time per frame: {runtime / len(dataset)}")
    
    ef, gls1, gls2 = results["ef"], results["basepoint_gls"], results["segmental_gls"]
    with open(os.path.join(args.visualize_dir, "ef_gls.txt"), "w") as f:
        f.write(f"EF: {ef}\n")
        f.write(f"GLS1: {gls1}")
        f.write(f"GLS2: {gls2}")
    print(f"Estimated EF value: {ef} - Estimated GLS value: {gls1} and {gls2}")
    
    results["dicom_path"] = args.dicom_path
    results["frame_start"] = frame_start_EF_GLS
    with open(os.path.join(args.visualize_dir, "results.json"), "w") as f:
        json.dump(results, f)
    
    now = datetime.datetime.now()
    df = pd.DataFrame({
        "date": [str(now.date())],
        "time": [str(now.time())],
        "case_idx": [case_idx],
        "case_name": [case_name],
        "dicom_path": [args.dicom_path],
        "ef": [ef], 
        "contour_gls": [gls1], 
        "segmental_gls": [gls2]
    })
    
    csv_output_dir = os.path.dirname(args.csv_output_file)
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)
    if os.path.exists(args.csv_output_file):
        df.to_csv(args.csv_output_file, mode="a", header=None, index=None)
    else:
        df.to_csv(args.csv_output_file, mode="a", index=None)
