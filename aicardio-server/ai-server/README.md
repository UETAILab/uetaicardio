# General
### Train segmentators
```
cd ~/pipeline
export PYTHONPATH=$(pwd)

# train full frame
python segmentation/trainers/train_single_frame.py --module_name segmentation --model_class UNet --epochs 40 --trainset /data.local/tuannm/data/2020-02-04_2C/train_dev --testset /data.local/tuannm/data/2020-02-04_2C/test --model_prefix tmp/giangunet --device "cuda:0"

# train cropped frame
python segmentation/trainers/train_single_frame.py --module_name segmentation --model_class UNet --epochs 40 --trainset /data.local/tuannm/data/2020-02-04_2C/video_20200223_cropped_train_dev --testset /data.local/tuannm/data/2020-02-04_2C/video_20200223_cropped_test --model_prefix tmp/cropped_giangunet --device "cuda:1"

# train full frame
python segmentation/trainers/train_single_frame_aux_loss.py --module_name segmentation --model_class AuxUNet --epochs 40 --trainset /data.local/tuannm/data/2020-02-04_2C/train_dev --testset /data.local/tuannm/data/2020-02-04_2C/test --model_prefix tmp/giangunet --device "cuda:0"

python segmentation/trainers/train_single_frame_aux_loss.py --module_name segmentation --model_class Resnet101DeeplabV3 --epochs 40 --trainset /data.local/tuannm/data/2020-02-04_2C/train_dev --testset /data.local/tuannm/data/2020-02-04_2C/test --model_prefix tmp/giangunet --device "cuda:0"

# train cropped frame
python segmentation/trainers/train_single_frame_aux_loss.py --module_name segmentation --model_class MegaAuxUNet --epochs 40 --trainset /data.local/tuannm/data/2020-02-04_2C/video_20200223_cropped_train_dev --testset /data.local/tuannm/data/2020-02-04_2C/video_20200223_cropped_test --model_prefix tmp/cropped_aux_hd_giangunet_invhd --device "cuda:1"

python segmentation/trainers/train_single_frame_aux_loss.py --module_name segmentation --model_class Resnet101DeeplabV3 --epochs 40 --trainset /data.local/tuannm/data/2020-02-04_2C/video_20200223_cropped_train_dev --testset /data.local/tuannm/data/2020-02-04_2C/video_20200223_cropped_test --model_prefix tmp/cropped_aux_hd_resnet101deeplabv3 --device "cuda:0"

```

### Problems
**Problems**.
* Classifier is sensitive to background details
* Must accept single-frame inputs

**TODO**.
* Add `is_valid` to items in all steps
    - [x] Reader
    - [x] Classifier
    - [x] Detector
    - [x] Segmentator
    - [x] Coarse analyzer
    - [x] Fine analyzer
    - [x] Pivot extractor
    - [x] Coeff estimator
* Test pipeline for running hints
    - [x] Pass ordinary DICOM test
    - [ ] Pass none-of-above DICOM test
* Fix GLS is nan problem
    * File: /data.local/data/DICOM_DATA/BVE/1.2.840.113663.1500.1.467297889.1.1.20200626.94435.911/IM_0174
    * Config: SPEED_DIRECTED_CONFIG
* Write readme and refactor code
    - [x] Refactor code
    - [x] Comment code
    - [ ] Write readme

# Inferencer
**Run inference pipeline**.
1. `cd ~/pipeline`
2. `export PYTHONPATH=$(pwd):$(pwd)/PyTorch_YOLOv3:$PYTHONPATH`
3. `conda activate /data.local/giangh/envs/pipeline`
4. `python inference/inferencer.py --dicom_path <path to dicom file> --out_dir <path to output dir>`

**Options to run inference pipeline**.
* Chamber classifier configuration
    * `classification_module_name` is the module, from which we import the chamber classification model. By default, this module is named `classification`. To check out chamber classification models implemented in this module, see `~/pipeline/classification/__init__.py`
    * `chamber_classifier` is the model class to be imported from `classification_module_name`. For example, if we use `classification_module_name` as `classification`, we have two models `MobilenetV2` (single frame) and `MultiframeClassifier` (multiple frames)
    * `classifier_weights` is the path to the `.pth` file containing the state_dict to be loaded into the model. The state dict must contain a key `model`, where the model weights are stored. See method `__load_model()` in file `~/pipeline/inference/classifiers/classifier.py` To see how model weights are loaded from the state dict
    * `classifier_batch_size` is the batch size used to run single-frame chamber classifier
* LV detector configuration
    * `detector_model_def` is the path to `.cfg` file specifying model architecture of YOLOv3 LV detector. We should only use the default path written in `~/pipeline/inference/inferencer.py`
    * `detector_model_weights` is the path to the `.pth` file containing the weights of YOLOv3 detector
    * `detector_batch_size` is the batch size used to run YOLOv3 LV detector
* LV segmentator configuration
    * `segmentator_module_name` is similar to `classification_module_name` but for LV segmentation models
    * Full-frame segmentator configuration
        * `full_frame_segmenatator` is similar to `chamber_classifier` but for full-frame LV segmentation models
        * `full_frame_weights` is similar to `classifier_weights` but for full-frame segmentation models
    * Cropped-frame segmentator configuration
        * `cropped_frame_segmentator` is similar to `chamber_classifier` but for cropped-frame LV segmentation models
        * `cropped_frame__weights` is similar to `classifier_weights` but for cropped-frame segmentation models
* `device` is the device, upon which the pipeline will operate, e.g. `"cuda:0"`

**Other configurations**. There are several other configurations related to algorithms implemented within the pipeline. You can check them out at file `~/pipeline/inference/config.py`
