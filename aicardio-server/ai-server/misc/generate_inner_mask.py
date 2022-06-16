import os
import glob
import argparse
import numpy as np
import cv2

def extract_inner_lv(data_root, save_dir):
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "masks"), exist_ok=True)

    mask_paths = glob.glob(os.path.join(data_root, "masks", "*.png"))
    mask_ids = [get_mask_id(mask_path) for mask_path in mask_paths]
    image_paths = [os.path.join(data_root, "images", f"{mask_id}.jpg") for mask_id in mask_ids]

    for mask_id, image_path, mask_path in zip(mask_ids, image_paths, mask_paths):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        binary_mask = convert_mask_to_binary(mask)

        inner_lv_area = image * binary_mask[..., None]
        cv2.imwrite(os.path.join(save_dir, "images", f"{mask_id}.jpg"), inner_lv_area)
        cv2.imwrite(os.path.join(save_dir, "masks", f"{mask_id}.png"), mask)

def get_mask_id(path):
    r"""Get mask ID from mask path"""
    mask_id = os.path.splitext(os.path.basename(path))[0]
    return mask_id

def convert_mask_to_binary(mask):
    r"""Convert [1..255] mask to 0-1 binary mask"""
    binary_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    binary_mask[mask[..., 0] >= 128] = 1
    return binary_mask

if __name__ == "__main__":
    # cd ~/pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainset", type=str, required=True, help="train_dev dataset folder")
    parser.add_argument("--testset", type=str, required=True, help="test dataset folder")
    parser.add_argument("--save_dir", type=str, required=True, help="directory to save extracted inner LV areas")
    args = parser.parse_args()

    extract_inner_lv(args.trainset, os.path.join(args.save_dir, "train"))
    extract_inner_lv(args.testset, os.path.join(args.save_dir, "test"))