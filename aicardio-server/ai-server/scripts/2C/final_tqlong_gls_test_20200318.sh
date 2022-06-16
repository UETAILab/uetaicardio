
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

VISUALIZE_DIR=${1:-"visualize-gls-20200326"}
CSV_OUTPUT_FILE=${4-"results/gls_20200326_resnet101_1.csv"}

DETECTOR_MODEL_WEIGHTS=${2-"/data.local/data/models/YOLO-2C/0.7435_yolov3_ckpt_75.pth"}

FULL_IMAGE_MODULE_NAME=${3-"segmentation"}
FULL_IMAGE_SEGMENTATION_CLASS=${4-"AuxUNet"}
FULL_IMAGE_SEGMENTATION_WEIGHTS=${5-"/data.local/data/models/segmentation/full2C/full_aux_giangunet_invhd_0009_0.8604_best.pth"}

CROPPED_IMAGE_MODULE_NAME=${3-"segmentation"}
CROPPED_IMAGE_SEGMENTATION_CLASS=${4-"Resnet101DeeplabV3"}
# CROPPED_IMAGE_SEGMENTATION_WEIGHTS=${5-"tmp/cropped_aux_hd_resnet101_b8_vloss_0011_0.8662_best.pth"} 
CROPPED_IMAGE_SEGMENTATION_WEIGHTS=${5-"/data.local/data/models/segmentation/cropped2C/cropped_aux_hd_resnet101_b8_vloss_0079_0.8707_best.pth"} # ??? WHY RUN THIS ON 17/03/2020

VISUALIZE_DIR=$(realpath $VISUALIZE_DIR)
echo "Results will be saved into ${VISUALIZE_DIR}"

declare -a DICOM_PATHS=(
                       "/data.local/tuannm/data/test_2020_03_13/GLS/0001__1.2.840.113663.1500.393838829248544842578148983107292161/IM_0006-2C" 
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0002__1.2.840.113663.1500.358022664305944126532169237319241277/IM_0028-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0004__1.2.840.113663.1500.920322511910100150329631783075791678/IM_0088-2C" #(*)
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0005__1.2.840.113663.1500.446679505129752076458377522253536361/IM_0117-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0007__1.2.840.113663.1500.651516284438297488088785836482616390/IM_0172-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0008__1.2.840.113663.1500.758623056237762615620230205666691082/IM_0204-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0009__1.2.840.113663.1500.792595419122355133601445980337272255/IM_0269-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0010__1.2.840.113663.1500.561668789166314219878396485567359266/IM_0290-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0011__1.2.840.113663.1500.278424863058641641953951093209094795/IM_0333-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0012__1.2.840.113663.1500.401111538248687666117266344497664058/IM_0359-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0013__1.2.840.113663.1500.807040638400593312164565033921895766/IM_0382-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0014__1.2.840.113663.1500.929574815420183481020171010502146253/IM_0417-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0016__1.2.840.113663.1500.309074879482106662567692642414504445/IM_0473-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0017__1.2.840.113663.1500.459267439567363483490791609265928739/IM_0498-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0018__1.2.840.113663.1500.331049636120831624031671910139680996/IM_0541-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0019__1.2.840.113663.1500.1.467297889.1.1.20191007.34600.782/IM_0051-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0021__1.2.840.113663.1500.1.458614274.1.1.20191025.50026.617/IM_0048-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0022__1.2.840.113663.1500.1.433991309.1.1.20180921.112034.560/IM_0004-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0023__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318732026.2/IM_0039-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0024__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318853881.2/IM_0036-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0025__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318945678.2/IM_0047-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0026__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319020281.2/IM_0050-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0027__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319102384.2/IM_0053-2C" #(*)
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0028__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319186863/IM_0033-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0029__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319246909.2/IM_0027-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0030__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319323079.2/IM_0024-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0031__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319399135.2/IM_0030-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0032__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319468164.2/IM_0043-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0033__1.2.840.113663.1500.1.458614274.1.1.20191111.225239.887/IM_0842-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0034__1.2.840.113663.1500.1.458614274.1.1.20191111.230933.101/IM_0758-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0035__1.2.840.113663.1500.1.458614274.1.1.20191111.231621.55/IM_0752-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0036__1.2.840.113663.1500.1.458614274.1.1.20191112.44903.699/IM_0747-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0037__1.2.840.113663.1500.1.458614274.1.1.20191017.221038.551/IM_0251-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0038__1.2.840.113663.1500.1.458614274.1.1.20191017.225530.742/IM_0238-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0039__1.2.840.113663.1500.1.458614274.1.1.20191018.5300.478/IM_0003-2C" #(*)
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0040__1.2.840.113663.1500.1.458614274.1.1.20191017.223259.830/IM_0002-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0041__1.2.840.113663.1500.1.458614274.1.1.20191017.220232.949/IM_0005-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0042__1.2.840.113663.1500.1.458614274.1.1.20191018.45727.933/IM_0197-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0043__1.2.840.113663.1500.1.458614274.1.1.20191018.40359.352/IM_0204-2C" # WRONG FRAME START
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0044__1.2.840.113663.1500.1.458614274.1.1.20191017.235431.126/IM_0227-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0045__1.2.840.113663.1500.1.458614274.1.1.20191017.232324.498/IM_0231-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0046__1.2.840.113663.1500.1.458614274.1.1.20191018.35306.94/IM_0219-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0047__1.2.840.113663.1500.1.458614274.1.1.20191018.50220.622/IM_0009-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0048__1.2.840.113663.1500.1.458614274.1.1.20191018.50939.643/IM_0188-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0049__1.2.840.113663.1500.1.458614274.1.1.20191031.225800.304/IM_0014-2C" #(*)
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0050__1.2.840.113663.1500.1.458614274.1.1.20191101.4539.285/IM_0015-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0051__1.2.840.113663.1500.1.458614274.1.1.20191101.41421.847/IM_0570-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0052__1.2.840.113663.1500.1.458614274.1.1.20191101.50041.610/IM_0019-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0053__1.2.840.113663.1500.1.458614274.1.1.20191101.45811.58/IM_0023-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0054__1.2.840.113663.1500.1.458614274.1.1.20191101.10049.24/IM_0026-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0055__1.2.840.113663.1500.1.458614274.1.1.20191101.11352.913/IM_0029-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0056__1.2.840.113663.1500.1.458614274.1.1.20191031.222703.77/IM_0634-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0057__1.2.840.113663.1500.1.458614274.1.1.20191031.233643.689/IM_0034-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0058__1.2.840.113663.1500.1.458614274.1.1.20191031.235055.3/IM_0035-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0059__1.2.840.113663.1500.1.458614274.1.1.20191101.12549.394/IM_0039-2C"
                        "/data.local/tuannm/data/test_2020_03_13/GLS/0060__1.2.840.113663.1500.1.458614274.1.1.20191031.223506.90/IM_0042-2C"
                        "/data.local/tuannm/data/test_2020_03_13/EF/0061__1.2.840.113663.1500.1.458614274.1.1.20191101.2100.372/IM_0595-2C"
                        "/data.local/tuannm/data/test_2020_03_13/EF/0062__1.2.840.113663.1500.1.458614274.1.1.20191101.42219.926/IM_0568-2C"
                        "/data.local/tuannm/data/test_2020_03_13/EF/0063__1.2.840.113663.1500.1.458614274.1.1.20191031.231254.761/IM_0609-2C"
                        "/data.local/tuannm/data/test_2020_03_13/EF/0064__1.2.840.113663.1500.1.458614274.1.1.20191018.34705.668/IM_0045-2C"
                        "/data.local/data/test_2020_03_13/EF/0065__STE_AN/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0066__STE_NAM/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0067__STE_OANH/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0068__STE_SON/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0069__STE_THAM/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0070__VAN.QUANG/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0071__DEID-0725--4529/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0072__DEID-1774--8182/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0073__DEID-2644--7025/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0074__DEID-0235--6277/IM_0002_2C"
                        "/data.local/data/test_2020_03_13/EF/0075__DEID-5637--0837/IM_0005_2C"
                        "/data.local/data/test_2020_03_13/EF/0076__DEID-1596--2245/IM_0002_2C"
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
