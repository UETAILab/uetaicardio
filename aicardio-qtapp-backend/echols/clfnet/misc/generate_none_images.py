import os
import glob
import tqdm

import pandas as pd
import numpy as np
import cv2

import pydicom as dcm


METADATA = "/data.local/giangh/pipeline/classification/datasets/metadata.csv"
OUTPUT_DIR = "/data.local/giangh/pipeline/data/classification/"
TRAIN_PROP = 0.8


def run():
    metadata = load_metadata()

    pbar = tqdm.tqdm(metadata.iterrows(), total=len(metadata))
    for i, row in pbar: 
        dataset = dcm.read_file(row.path)
        frames = dataset.pixel_array
        save(row, frames)

def load_metadata():
    metadata = pd.read_csv(METADATA)
    metadata = metadata[metadata.is_none_class == True]
    return metadata

def save(metadata, frames):
    _dir, basename = os.path.split(metadata.path)
    _dir, casename = os.path.split(_dir)
    outpath = os.path.join(OUTPUT_DIR, metadata.split, "none", f"{basename}_{casename}.jpg")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    cv2.imwrite(outpath, frames)


if __name__ == "__main__":
    run()
