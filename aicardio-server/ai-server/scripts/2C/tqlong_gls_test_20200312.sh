# Test with Thanh's DICOM files for GLS

# cd ~/pipeline
# export PYTHONPATH=$(pwd):$(pwd)/PyTorch_YOLOv3
# bash bash_scripts/tqlong_ef_test.sh [VISUALIZE_DIR] [DETECTOR_MODEL_WEIGHTS] \
#                                     [FULL_IMAGE_MODULE_NAME] [FULL_IMAGE_SEGMENTATION_CLASS] \
#              c                       [FULL_IMAGE_SEGMENTATION_WEIGHTS] \
#                                     [CROPPED_IMAGE_MODULE_NAME] [CROPPED_IMAGE_SEGMENTATION_CLASS] \
#                                     [CROPPED_IMAGE_SEGMENTATION_WEIGHTS]
# NOTE: `[]` means optional

VISUALIZE_DIR=${1:-"visualize"}
CSV_OUTPUT_FILE=${4-"tmp/ef_gls_results.csv"}
DETECTOR_MODEL_WEIGHTS=${2-"/data.local/data/models/YOLO-2C/0.7435_yolov3_ckpt_75.pth"}

FULL_IMAGE_MODULE_NAME=${3-"segmentation"}
FULL_IMAGE_SEGMENTATION_CLASS=${4-"AuxUNet"}
FULL_IMAGE_SEGMENTATION_WEIGHTS=${5-"/data.local/data/models/segmentation/full2C/full_aux_giangunet_invhd_0009_0.8604_best.pth"}

CROPPED_IMAGE_MODULE_NAME=${3-"segmentation"}
CROPPED_IMAGE_SEGMENTATION_CLASS=${4-"AuxUNet"}
CROPPED_IMAGE_SEGMENTATION_WEIGHTS=${5-"/data.local/data/models/segmentation/cropped2C/cropped_aux_bce_invhd_giangunet_0014_0.8642_best.pth"}

VISUALIZE_DIR=$(realpath $VISUALIZE_DIR)
echo "Results will be saved into ${VISUALIZE_DIR}"

declare -a DICOM_PATHS=("/data.local/tuannm/data/test_2020_03_11/STE QUY ANALYZED/GLS/IM_0034-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191101.41421.847/GLS/IM_0570-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319468164.2/GLS/IM_0043-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.433991309.1.1.20180921.112034.560/GLS/IM_0004-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191031.222703.77/GLS/IM_0634-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319399135.2/GLS/IM_0030-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE HAI analyzed/GLS/IM_0019-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE PHUONG ANALYZED/GLS/IM_0005-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191101.42219.926/GLS/IM_0568-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191018.35306.94/GLS/IM_0219-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE CAU ANALYZED/GLS/IM_0015-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE PHI ANALYZED/GLS/IM_0002-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191101.2100.372/GLS/IM_0595-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191017.225530.742/GLS/IM_0238-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE SON ANALYZED/GLS/IM_0035-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE SON  ID 1101 ANALYZED/GLS/IM_0039-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE NGA ANALYZED/GLS/IM_0029-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE LAI ANALYZED/GLS/IM_0023-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319102384.2/GLS/IM_0053-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191111.230933.101/GLS/IM_0758-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191031.231254.761/GLS/IM_0609-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE BAU ANALYZED/GLS/IM_0014-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318732026.2/GLS/IM_0039-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191017.232324.498/GLS/IM_0231-2C"
                        "/data.local/tuannm/data/test_2020_03_11/VD TEST CASE 1 TRUONG ANALYZED/GLS/IM_0048-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319186863/GLS/IM_0033-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319323079.2/GLS/IM_0024-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE CHINH FOLDER VD1 ANALYZED/GLS/IM_0045-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191112.44903.699/GLS/IM_0747-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319246909.2/GLS/IM_0027-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318945678.2/GLS/IM_0047-2C"
                        "/data.local/tuannm/data/test_2020_03_11/VIET ANALYZED/GLS/IM_0009-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191111.231621.55/GLS/IM_0752-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE MAI ANALYZED/GLS/IM_0026-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319020281.2/GLS/IM_0050-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191018.50939.643/GLS/IM_0188-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191018.5300.478/GLS/IM_0003-2C"
                        "/data.local/tuannm/data/test_2020_03_11/BV E TEST CASE 1 ANALYZED/GLS/IM_0051-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191017.221038.551/GLS/IM_0251-2C"
                        "/data.local/tuannm/data/test_2020_03_11/STE THE ANALYZED/GLS/IM_0042-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318853881.2/GLS/IM_0036-2C"
                        "/data.local/tuannm/data/test_2020_03_11/1.2.840.113663.1500.1.458614274.1.1.20191111.225239.887/GLS/IM_0842-2C")

for DICOM_PATH in "${DICOM_PATHS[@]}"
do
    DCM_PATH=${DICOM_PATH//"/data.local/tuannm/data/test_2020_03_11/"/};
    DCM_PATH=${DCM_PATH//"/GLS"/};
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