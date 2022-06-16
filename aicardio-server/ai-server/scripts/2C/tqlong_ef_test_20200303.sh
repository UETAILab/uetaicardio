# Test with tqlong's DICOM files

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

declare -a DICOM_PATHS=("/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191101.41421.847/IM_0570-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.433991309.1.1.20180921.112034.560/IM_0004-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191031.222703.77/IM_0634-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319399135.2/IM_0030-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191101.42219.926/IM_0567-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191018.35306.94/IM_0219-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191101.2100.372/IM_0595-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191017.225530.742/IM_0237-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.428468407.1.1.20200213.85129.81/IM_0144-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.428468407.1.1.20200106.144204.860/IM_0045-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.428468407.1.1.20200106.151800.807/IM_0126-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191025.50026.617/IM_0005-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319102384.2/IM_0053-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191111.230933.101/IM_0758-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191031.231254.761/IM_0609-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191018.40359.352/IM_0204-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318732026.2/IM_0039-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191017.232324.498/IM_0231-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319186863/IM_0033-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191031.233643.689/IM_0606-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319323079.2/IM_0024-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191101.4539.285/IM_0590-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319246909.2/IM_0027-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318945678.2/IM_0047-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.428468407.1.1.20200106.150846.22/IM_0099-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319020281.2/IM_0050-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191101.45811.58/IM_0642-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191018.50939.643/IM_0188-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191017.220232.949/IM_0256-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191018.5300.478/IM_0003-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191101.50041.610/IM_0646-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191031.225800.304/IM_0615-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191018.50220.622/IM_0192-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191017.221038.551/IM_0251-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191031.235055.3/IM_0599-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318853881.2/IM_0036-2C"
                        "/data.local/tuannm/data/dicom234_2020_03_03/1.2.840.113663.1500.1.458614274.1.1.20191111.225239.887/IM_0843-2C"))

for DICOM_PATH in "${DICOM_PATHS[@]}"
do
    DCM_PATH=${DICOM_PATH//"/data.local/tuannm/data/dicom234_2020_03_03/"/};
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