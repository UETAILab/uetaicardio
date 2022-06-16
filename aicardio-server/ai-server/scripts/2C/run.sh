
# Final test
# NOTE: ko co gi thi lay chu ki dau tien, tu frame 0, voi +10 or +15% window

# cd ~/pipeline
# export PYTHONPATH=$(pwd):$(pwd)/PyTorch_YOLOv3
# bash bash_scripts/tqlong_ef_test.sh [VISUALIZE_DIR] [DETECTOR_MODEL_WEIGHTS] \
#              [FULL_IMAGE_MODULE_NAME] [FULL_IMAGE_SEGMENTATION_CLASS] \
#              [FULL_IMAGE_SEGMENTATION_WEIGHTS] \
#              [CROPPED_IMAGE_MODULE_NAME] [CROPPED_IMAGE_SEGMENTATION_CLASS] \
#              [CROPPED_IMAGE_SEGMENTATION_WEIGHTS]
# NOTE: `[]` means optional
DICOM_PATH=${1}

VISUALIZE_DIR=${2:-"results/visualization/visualize-gls-2C-20200428"}
CSV_OUTPUT_FILE=${3-"results/estimation/gls_2C_20200428_resnet101_1.csv"}

DETECTOR_MODEL_WEIGHTS=${4-"/data.local/data/models/YOLO-2C/0.7435_yolov3_ckpt_75.pth"}

FULL_IMAGE_MODULE_NAME=${5-"segmentation"}
FULL_IMAGE_SEGMENTATION_CLASS=${6-"AuxUNet"}
FULL_IMAGE_SEGMENTATION_WEIGHTS=${7-"/data.local/data/models/segmentation/full2C/full_aux_giangunet_invhd_0009_0.8604_best.pth"}

CROPPED_IMAGE_MODULE_NAME=${8-"segmentation"}
CROPPED_IMAGE_SEGMENTATION_CLASS=${9-"Resnet101DeeplabV3"}
# CROPPED_IMAGE_SEGMENTATION_WEIGHTS=${5-"tmp/cropped_aux_hd_resnet101_b8_vloss_0011_0.8662_best.pth"} 
CROPPED_IMAGE_SEGMENTATION_WEIGHTS=${10-"/data.local/data/models/segmentation/cropped2C/cropped_aux_hd_resnet101_b8_vloss_0079_0.8707_best.pth"} # ??? WHY RUN THIS ON 17/03/2020

FRAME_START_FILE=${11-"/data.local/giangh/pipeline/scripts/2C/frame_start_EF_GLS.json"}

VISUALIZE_DIR=$(realpath $VISUALIZE_DIR)
echo "Results will be saved into ${VISUALIZE_DIR}"

python lv_inference/lv_pipeline.py --detector_model_weights "$DETECTOR_MODEL_WEIGHTS" \
                --full_image_module_name "$FULL_IMAGE_MODULE_NAME" \
                --full_image_segmentation_class "$FULL_IMAGE_SEGMENTATION_CLASS" \
                --full_image_segmentation_weights "$FULL_IMAGE_SEGMENTATION_WEIGHTS" \
                --cropped_image_module_name "$CROPPED_IMAGE_MODULE_NAME" \
                --cropped_image_segmentation_class "$CROPPED_IMAGE_SEGMENTATION_CLASS" \
                --cropped_image_segmentation_weights "$CROPPED_IMAGE_SEGMENTATION_WEIGHTS" \
                --dicom_path "$DICOM_PATH" \
                --visualize_dir "${VISUALIZE_DIR}/${DCM_PATH}" \
                --csv_output_file "$CSV_OUTPUT_FILE" \
                --frame_start_json_file "$FRAME_START_FILE" \
                --chamber "2C"