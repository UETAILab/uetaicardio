import os
import glob
import tqdm
import pydicom as dcm

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


ROOT = "/data.local/data/DICOM_DATA"
HOSPITALS = [os.path.basename(path) for path in glob.glob(os.path.join(ROOT, "*"))]
METADATA = []
for hospital in HOSPITALS:
    paths = [path for path in glob.glob(os.path.join(ROOT, hospital, "*", "*"))
             if ".txt" not in path]
    METADATA.extend([{"path": path, "hospital": hospital, "is_valid": np.nan, "shape": np.nan}
                     for path in paths])
METADATA = pd.DataFrame(METADATA)


def main():
    pbar = tqdm.tqdm(METADATA.iterrows(), total=len(METADATA))
    for i, item in pbar:
        try:
            dataset = dcm.read_file(item["path"])
            frames = dataset.pixel_array
        except Exception as e:
            METADATA.loc[i, "is_valid"] = False
        else:
            METADATA.loc[i, "is_valid"] = True
            METADATA.loc[i, "shape"] = str(frames.shape)[1:-1]

        if i%100 == 0:
            METADATA.to_csv("/data.local/giangh/pipeline/classification/datasets/metadata.csv")
    METADATA.to_csv("/data.local/giangh/pipeline/classification/datasets/metadata.csv")


if __name__ == "__main__":
    main()
