#!/bin/bash
# conda activate /data.local/giangh/envs/pipeline before running commands
# Usage: ./scripts/functions/chamber_classifier.sh [MODE] [PATH] \
#                                                  [PIPELINE CONFIG] \
#                                                  [OUT_DIR]
# [MODE] is whether the input is DICOM dir or DICOM file
#   [MODE] can be "dir" or "file"
# [PATH] is path to DICOM dir or DICOM path, depending on [MODE]
# [PIPELINE CONFIG] is path to pipeline config JSON file
# [OUT_DIR] is directory to save results

export PYTHONPATH=$(pwd):$(pwd)/PyTorch_YOLOv3:$PYTHONPATH

MODE=${1-"file"}
PATH=${2}
PIPELINE_CONFIG=${3-"/data.local/giangh/pipeline/inference/functions/hint_generator_pipeline_config/2C_example.json"}
OUTDIR=${4-"tmp"}

if [ "$MODE" == "dir" ]; then
    ./inference/functions/chamber_classifier.py \
        --dicom_dir "$PATH" \
        --pipeline_config "$PIPELINE_CONFIG" \
        --out_dir "$OUTDIR"
elif [ "$MODE" == "file" ]; then
    ./inference/functions/chamber_classifier.py 
        --dicom_file "$PATH" \
        --pipeline_config "$PIPELINE_CONFIG" \
        --out_dir "$OUTDIR"
else
    echo "Invalid MODE, must be 'dir' or 'file'"
 fi
