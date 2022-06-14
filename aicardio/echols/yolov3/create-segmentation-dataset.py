from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def generate_dataset(src_dir, tgt_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(os.path.join(tgt_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(tgt_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(tgt_dir, "metadata"), exist_ok=True)
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights_path))
    model.eval()

    dataloader = DataLoader(
        ImageFolder(os.path.join(opt.src_dir,"images"), img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    classes = load_classes(opt.class_path)
    
    predicts = []
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # print(img_paths)
        input_imgs = Variable(input_imgs.type(Tensor)).to(device)

        # Get detections
        with torch.no_grad():
            img_detections = model(input_imgs)
            img_detections = non_max_suppression(img_detections, opt.conf_thres, opt.nms_thres)

            for (path, detections) in zip(img_paths, img_detections):
                msk_path = os.path.join(os.path.dirname(path), "..", "masks", f"{os.path.basename(path)[:-4]}.png")
                if not os.path.exists(msk_path): continue
                
                img = np.array(Image.open(path))
                msk = np.array(Image.open(msk_path))
                
#                 print(path)
                if detections is not None and len(detections) == 1:
                    detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                    
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        width, height = x2-x1, y2-y1
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # print("y1", x1, y1, x2, y2)
                        y1 = max(0, int(y1 - 0.75*(y2-y1)))
                        y2 = min(img.shape[0], int(y2 + 0.25*(y2-y1)))
                        x1 = max(0, int(x1 - 0.25*(x2-x1)))
                        x2 = min(img.shape[1], int(x2 + 0.25*(x2-x1)))
                        # print("y1", x1, y1, x2, y2)
#                         bbox_img = img[y1:y2, x1:x2]
                        path_bn = os.path.basename(path)
                        dicom_fn = path_bn[:path_bn.rindex("_")]
                        print("dicom_fn:", path_bn)

                        bbox_img_path = os.path.join(tgt_dir, "images", os.path.basename(path))
#                         Image.fromarray(bbox_img).save(bbox_img_path)

#                         bbox_msk = msk[y1:y2, x1:x2]
                        bbox_msk_path = os.path.join(tgt_dir, "masks", f"{os.path.basename(path)[:-4]}.png")
#                         Image.fromarray(bbox_msk).save(bbox_msk_path)
                        
                        predicts.append([dicom_fn, bbox_img_path, bbox_msk_path, x1, y1, x2, y2, os.path.abspath(path), os.path.abspath(msk_path) ])
                        

    # group images in video together
    predicts.sort()
    predicts_dicts = {}
    for predict in predicts:
        dicom_fn = predict[0]
#         print("ori value: ", predict)
        if dicom_fn in predicts_dicts:
            old_value = predicts_dicts[dicom_fn]
#             print("old value: ", old_value)
            predict[3] = min(old_value[3], predict[3])
            predict[4] = min(old_value[4], predict[4])
            predict[5] = max(old_value[5], predict[5])
            predict[6] = max(old_value[6], predict[6])
        
        predicts_dicts[dicom_fn] = predict
#         print("new value: ", predict)

    # generate cropped images full frames
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor)).to(device)
        
        with torch.no_grad():
            for path in img_paths:
                
                img = np.array(Image.open(path))
                path_bn = os.path.basename(path)
                dicom_fn = path_bn[:path_bn.rindex("_")]
                # check dicom file in dicts
                if dicom_fn in predicts_dicts:
                    print("Frame: {}".format(path_bn))
                    pr_value = predicts_dicts[dicom_fn]
                    x1 = pr_value[3]
                    y1 = pr_value[4]
                    x2 = pr_value[5]
                    y2 = pr_value[6]
                    
                    bbox_img = img[y1:y2, x1:x2]
                    bbox_img_path = os.path.join(tgt_dir, "images", path_bn)
                    Image.fromarray(bbox_img).save(bbox_img_path)
                    
                    # mask
                    msk_path = os.path.join(os.path.dirname(path), "..", "masks", f"{path_bn[:-4]}.png")
                    if not os.path.exists(msk_path): continue
                    msk = np.array(Image.open(msk_path))
                    bbox_msk = msk[y1:y2, x1:x2]
                    bbox_msk_path = os.path.join(tgt_dir, "masks", f"{path_bn[:-4]}.png")
                    Image.fromarray(bbox_msk).save(bbox_msk_path)
                    
                    bbox_metadata_path = os.path.join(tgt_dir, "metadata", f"{path_bn[:-4]}.txt")
                    with open(bbox_metadata_path, "w") as f:
                        f.write(f"{x1} {y1} {x2} {y2}")

                
    
#     for predict in predicts:
#         dicom_fn = predict[0]
#         pr_value = predicts_dicts[dicom_fn]
        
#         x1 = pr_value[3]
#         y1 = pr_value[4]
#         x2 = pr_value[5]
#         y2 = pr_value[6]
        
#         path = predict[7]
#         msk_path = predict[8]
        
#         print("Frame: {}".format(predict[1]))
        
#         img = np.array(Image.open(path))
#         msk = np.array(Image.open(msk_path))
        

#         bbox_img = img[y1:y2, x1:x2]
#         bbox_img_path = predict[1]
#         Image.fromarray(bbox_img).save(bbox_img_path)

#         bbox_msk = msk[y1:y2, x1:x2]
#         bbox_msk_path = predict[2]
#         Image.fromarray(bbox_msk).save(bbox_msk_path)
        
#     import json        
#     with open("predicts_txt.json", "w") as fw:
        
#         json.dump(predicts, fw)
#         fw.close()
               
                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints-2020-02-04_2C/yolov3_ckpt_3.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="/data.local/data/YOLO-custom-folder/custom-2020-02-04_2C/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--src_dir", type=str, required=True, help="/data.local/tuannm/data/2020-02-04_2C/train_dev")
    parser.add_argument("--tgt_dir", type=str, required=True, help="/data.local/tuannm/data/2020-02-04_2C/cropped_train_dev")
    opt = parser.parse_args()
    print(opt)

    src_dir = opt.src_dir #"/data.local/tuannm/data/2020-02-04_2C/train_dev"
    tgt_dir = opt.tgt_dir # "/data.local/tuannm/data/2020-02-04_2C/cropped_train_dev"
    generate_dataset(src_dir, tgt_dir)

