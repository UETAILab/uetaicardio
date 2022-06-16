
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

VISUALIZE_DIR=${1:-"results/visualization/visualize-gls-20200402-3C"}
CSV_OUTPUT_FILE=${4-"results/estimation/3C_gls_20200402_resnet101_1.csv"}

DETECTOR_MODEL_WEIGHTS=${2-"/data.local/data/models/YOLO-3C/0.7725_yolov3_ckpt_96.pth"}

FULL_IMAGE_MODULE_NAME=${3-"segmentation"}
FULL_IMAGE_SEGMENTATION_CLASS=${4-"Resnet101DeeplabV3"}
FULL_IMAGE_SEGMENTATION_WEIGHTS=${5-"/data.local/giangh/pipeline/tmp/3C_full_aux_hd_resnet101deeplabv3_0058_0.8621_best.pth"}

CROPPED_IMAGE_MODULE_NAME=${3-"segmentation"}
CROPPED_IMAGE_SEGMENTATION_CLASS=${4-"Resnet101DeeplabV3"}
CROPPED_IMAGE_SEGMENTATION_WEIGHTS=${5-"/data.local/giangh/pipeline/tmp/3C_cropped_aux_hd_resnet101deeplabv3_0022_0.8688_best.pth"}

FRAME_START_FILE=${6-"/data.local/giangh/pipeline/bash_scripts/4C/frame_start_EF_GLS.json"}

VISUALIZE_DIR=$(realpath $VISUALIZE_DIR)
echo "Results will be saved into ${VISUALIZE_DIR}"

declare -a DICOM_PATHS=(
    "/data.local/data/GLS/0001__1.2.840.113663.1500.393838829248544842578148983107292161/IM_0008-3C"
    "/data.local/data/GLS/0002__1.2.840.113663.1500.358022664305944126532169237319241277/IM_0031-3C"
    "/data.local/data/GLS/0004__1.2.840.113663.1500.920322511910100150329631783075791678/IM_0089-3C"
    "/data.local/data/GLS/0005__1.2.840.113663.1500.446679505129752076458377522253536361/IM_0119-3C"
    "/data.local/data/GLS/0007__1.2.840.113663.1500.651516284438297488088785836482616390/IM_0177-3C"
    "/data.local/data/GLS/0008__1.2.840.113663.1500.758623056237762615620230205666691082/IM_0209-3C"
    "/data.local/data/GLS/0009__1.2.840.113663.1500.792595419122355133601445980337272255/IM_0272-3C"
    "/data.local/data/GLS/0010__1.2.840.113663.1500.561668789166314219878396485567359266/IM_0292-3C"
    "/data.local/data/GLS/0011__1.2.840.113663.1500.278424863058641641953951093209094795/IM_0335-3C"
    "/data.local/data/GLS/0012__1.2.840.113663.1500.401111538248687666117266344497664058/IM_0362-3C"
    "/data.local/data/GLS/0013__1.2.840.113663.1500.807040638400593312164565033921895766/IM_0385-3C"
    "/data.local/data/GLS/0014__1.2.840.113663.1500.929574815420183481020171010502146253/IM_0414-3C"
    "/data.local/data/GLS/0016__1.2.840.113663.1500.309074879482106662567692642414504445/IM_0475-3C"
    "/data.local/data/GLS/0017__1.2.840.113663.1500.459267439567363483490791609265928739/IM_0500-3C"
    "/data.local/data/GLS/0018__1.2.840.113663.1500.331049636120831624031671910139680996/IM_0544-3C"
    "/data.local/data/GLS/0019__1.2.840.113663.1500.1.467297889.1.1.20191007.34600.782/IM_0052-3C"
    "/data.local/data/GLS/0021__1.2.840.113663.1500.1.458614274.1.1.20191025.50026.617/IM_0049-3C"
    "/data.local/data/GLS/0022__1.2.840.113663.1500.1.433991309.1.1.20180921.112034.560/IM_0006-3C"
    "/data.local/data/GLS/0023__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318732026.2/IM_0040-3C"
    "/data.local/data/GLS/0024__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318853881.2/IM_0037-3C"
    "/data.local/data/GLS/0025__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519318945678.2/IM_0048-3C"
    "/data.local/data/GLS/0026__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319020281.2/IM_0051-3C"
    "/data.local/data/GLS/0027__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319102384.2/IM_0054-3C"
    "/data.local/data/GLS/0028__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319186863/IM_0034-3C"
    "/data.local/data/GLS/0029__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319246909.2/IM_0028-3C"
    "/data.local/data/GLS/0030__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319323079.2/IM_0025-3C"
    "/data.local/data/GLS/0031__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319399135.2/IM_0031-3C"
    "/data.local/data/GLS/0032__1.3.46.670589.14.20001.309.2.4.46.678.1.7488.1519319468164.2/IM_0044-3C"
    "/data.local/data/GLS/0033__1.2.840.113663.1500.1.458614274.1.1.20191111.225239.887/IM_0843-3C"
    "/data.local/data/GLS/0034__1.2.840.113663.1500.1.458614274.1.1.20191111.230933.101/IM_0759-3C"
    "/data.local/data/GLS/0035__1.2.840.113663.1500.1.458614274.1.1.20191111.231621.55/IM_0755-3C"
    "/data.local/data/GLS/0036__1.2.840.113663.1500.1.458614274.1.1.20191112.44903.699/IM_0748-3C"
    "/data.local/data/GLS/0037__1.2.840.113663.1500.1.458614274.1.1.20191017.221038.551/IM_0252-3C"
    "/data.local/data/GLS/0038__1.2.840.113663.1500.1.458614274.1.1.20191017.225530.742/IM_0237-3C"
    "/data.local/data/GLS/0039__1.2.840.113663.1500.1.458614274.1.1.20191018.5300.478/IM_0004-3C"
    "/data.local/data/GLS/0040__1.2.840.113663.1500.1.458614274.1.1.20191017.223259.830/IM_0003-3C"
    "/data.local/data/GLS/0041__1.2.840.113663.1500.1.458614274.1.1.20191017.220232.949/IM_0006-3C"
    "/data.local/data/GLS/0042__1.2.840.113663.1500.1.458614274.1.1.20191018.45727.933/IM_0199-3C"
    "/data.local/data/GLS/0043__1.2.840.113663.1500.1.458614274.1.1.20191018.40359.352/IM_0203-3C"
    "/data.local/data/GLS/0044__1.2.840.113663.1500.1.458614274.1.1.20191017.235431.126/IM_0011-3C"
    "/data.local/data/GLS/0044__1.2.840.113663.1500.1.458614274.1.1.20191017.235431.126/IM_0228-3C"
    "/data.local/data/GLS/0045__1.2.840.113663.1500.1.458614274.1.1.20191017.232324.498/IM_0232-3C"
    "/data.local/data/GLS/0046__1.2.840.113663.1500.1.458614274.1.1.20191018.35306.94/IM_0220-3C"
    "/data.local/data/GLS/0047__1.2.840.113663.1500.1.458614274.1.1.20191018.50220.622/IM_0008-3C"
    "/data.local/data/GLS/0048__1.2.840.113663.1500.1.458614274.1.1.20191018.50939.643/IM_0189-3C"
    "/data.local/data/GLS/0049__1.2.840.113663.1500.1.458614274.1.1.20191031.225800.304/IM_0013-3C"
    "/data.local/data/GLS/0050__1.2.840.113663.1500.1.458614274.1.1.20191101.4539.285/IM_0017-3C"
    "/data.local/data/GLS/0051__1.2.840.113663.1500.1.458614274.1.1.20191101.41421.847/IM_0571-3C"
    "/data.local/data/GLS/0052__1.2.840.113663.1500.1.458614274.1.1.20191101.50041.610/IM_0020-3C"
    "/data.local/data/GLS/0053__1.2.840.113663.1500.1.458614274.1.1.20191101.45811.58/IM_0021-3C"
    "/data.local/data/GLS/0054__1.2.840.113663.1500.1.458614274.1.1.20191101.10049.24/IM_0027-3C"
    "/data.local/data/GLS/0055__1.2.840.113663.1500.1.458614274.1.1.20191101.11352.913/IM_0030-3C"
    "/data.local/data/GLS/0056__1.2.840.113663.1500.1.458614274.1.1.20191031.222703.77/IM_0635-3C"
    "/data.local/data/GLS/0057__1.2.840.113663.1500.1.458614274.1.1.20191031.233643.689/IM_0033-3C"
    "/data.local/data/GLS/0058__1.2.840.113663.1500.1.458614274.1.1.20191031.235055.3/IM_0036-3C"
    "/data.local/data/GLS/0059__1.2.840.113663.1500.1.458614274.1.1.20191101.12549.394/IM_0040-3C"
    "/data.local/data/GLS/0060__1.2.840.113663.1500.1.458614274.1.1.20191031.223506.90/IM_0043-3C"
    "/data.local/data/GLS/0061__1.2.840.113663.1500.1.458614274.1.1.20191101.2100.372/IM_0596-3C"
    "/data.local/data/GLS/0062__1.2.840.113663.1500.1.458614274.1.1.20191101.42219.926/IM_0567-3C"
    "/data.local/data/GLS/0063__1.2.840.113663.1500.1.458614274.1.1.20191031.231254.761/IM_0610-3C"
    "/data.local/data/GLS/0064__1.2.840.113663.1500.1.458614274.1.1.20191018.34705.668/IM_0046-3C"
    "/data.local/data/GLS/0077__DEID 5495--4515/IM_0003-3C"
    "/data.local/data/GLS/0078__DEID 9607--7715/IM_0003-3C"
    "/data.local/data/GLS/0080__DEID 6688--2306/IM_0003-3C"
    "/data.local/data/GLS/0081__DEID 2085--2033/IM_0003-3C"
    "/data.local/data/GLS/0082__DEID 4638--8720/IM_0003-3C"
    "/data.local/data/GLS/0083__DEID 4684--7284/IM_0003-3C"
    "/data.local/data/GLS/0084__DEID 8522--8063/IM_0003-3C"
    "/data.local/data/GLS/0085__DEID 9722--9965/IM_0003-3C"
    "/data.local/data/GLS/0086__DEID 9514--6377/IM_0003-3C"
    "/data.local/data/GLS/0087__DEID 6439--8908/IM_0003-3C"
    "/data.local/data/GLS/0088__DEID 9282--7661/IM_0003-3C"
    "/data.local/data/GLS/0089__DEID 1447--2162/IM_0003-3C"
    "/data.local/data/GLS/0090__DEID 7556--1733/IM_0003-3C"
    "/data.local/data/GLS/0091__DEID 9539--4539/IM_0001-3C"
    "/data.local/data/GLS/0092__DEID 8992--6496/IM_0003-3C"
    "/data.local/data/GLS/0093__DEID 2641--6461/IM_0003-3C"
    "/data.local/data/GLS/0094__DEID 1384--8656/IM_0003-3C"
    "/data.local/data/GLS/0095__HA VAN LO 37Y/IM_0003-3C"
    "/data.local/data/GLS/0096__AI.2522020.THUY60Y/IM_0003-3C"
    "/data.local/data/GLS/0097__DEID 9407--4673/IM_0003-3C"
    "/data.local/data/GLS/0098__DEID 9223--0901/IM_0003-3C"
    "/data.local/data/GLS/0099__DEID 8906--4303/IM_0003-3C"
    "/data.local/data/GLS/0100__DEID 8896--3907/IM_0003-3C"
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
                --csv_output_file "$CSV_OUTPUT_FILE" \
                --frame_start_json_file "$FRAME_START_FILE" \
                --chamber "3C"
done
