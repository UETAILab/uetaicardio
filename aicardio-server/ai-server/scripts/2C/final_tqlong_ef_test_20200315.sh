
# Final test
# NOTE: ko co gi thi lay chu ki dau tien, tu frame 0, voi +10 or +15% window

# cd ~/pipeline
# export PYTHONPATH=$(pwd):$(pwd)/PyTorch_YOLOv3
# bash bash_scripts/tqlong_ef_test.sh [VISUALIZE_DIR] [DETECTOR_MODEL_WEIGHTS] \
#                                     [FULL_IMAGE_MODULE_NAME] [FULL_IMAGE_SEGMENTATION_CLASS] \
#                                     [FULL_IMAGE_SEGMENTATION_WEIGHTS] \
#                                     [CROPPED_IMAGE_MODULE_NAME] [CROPPED_IMAGE_SEGMENTATION_CLASS] \
#                                     [CROPPED_IMAGE_SEGMENTATION_WEIGHTS]
# NOTE: `[]` means optional

VISUALIZE_DIR=${1:-"visualize"}
CSV_OUTPUT_FILE=${4-"results/ef_20200315_resnet101.csv"}
DETECTOR_MODEL_WEIGHTS=${2-"/data.local/data/models/YOLO-2C/0.7435_yolov3_ckpt_75.pth"}

FULL_IMAGE_MODULE_NAME=${3-"segmentation"}
FULL_IMAGE_SEGMENTATION_CLASS=${4-"AuxUNet"}
FULL_IMAGE_SEGMENTATION_WEIGHTS=${5-"/data.local/data/models/segmentation/full2C/full_aux_giangunet_invhd_0009_0.8604_best.pth"}

CROPPED_IMAGE_MODULE_NAME=${3-"segmentation"}
CROPPED_IMAGE_SEGMENTATION_CLASS=${4-"Resnet101DeeplabV3"}
CROPPED_IMAGE_SEGMENTATION_WEIGHTS=${5-"tmp/cropped_aux_hd_resnet101_b8_vloss_0079_0.8707_best.pth"}

VISUALIZE_DIR=$(realpath $VISUALIZE_DIR)
echo "Results will be saved into ${VISUALIZE_DIR}"

declare -a DICOM_PATHS=(
/data.local/data/EF/0065__STE_AN/IM_0002_2C
/data.local/data/EF/0067__STE_OANH/IM_0002_2C
/data.local/data/EF/0073__DEID-2644--7025/IM_0002_2C
/data.local/data/EF/0070__VAN.QUANG/IM_0002_2C
/data.local/data/EF/0068__STE_SON/IM_0002_2C
/data.local/data/EF/0069__STE_THAM/IM_0002_2C
/data.local/data/EF/0071__DEID-0725--4529/IM_0002_2C
/data.local/data/EF/0074__DEID-0235--6277/IM_0002_2C
/data.local/data/EF/0075__DEID-5637--0837/IM_0005_2C
/data.local/data/EF/0072__DEID-1774--8182/IM_0002_2C
/data.local/data/EF/0066__STE_NAM/IM_0002_2C
/data.local/data/EF/0076__DEID-1596--2245/IM_0002_2C
			)

for DICOM_PATH in "${DICOM_PATHS[@]}"
do
    DCM_PATH=${DICOM_PATH//"/data.local/tuannm/data/test_2020_03_13/"/};
    python lv_inference/lv_pipeline.py --detector_model_weights "$DETECTOR_MODEL_WEIGHTS" \
                                       --full_image_module_name "$FULL_IMAGE_MODULE_NAME" \
                                       --full_image_segmentation_class "$FULL_IMAGE_SEGMENTATION_CLASS" \
                                       --full_image_segmentation_weights "$FULL_IMAGE_SEGMENTATION_WEIGHTS" \
                                       --cropped_image_module_name "$CROPPED_IMAGE_MODULE_NAME" \
                                       --cropped_image_segmentation_class "$CROPPED_IMAGE_SEGMENTATION_CLASS" \
                                       --cropped_image_segmentation_weights "$CROPPED_IMAGE_SEGMENTATION_WEIGHTS" \
                                       --dicom_path "$DICOM_PATH" \
                                       --visualize_dir "${VISUALIZE_DIR}/${DCM_PATH}" \
                                       --csv_output_file "$CSV_OUTPUT_FILE"
done
