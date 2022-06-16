#!/bin/bash

MODE=${1}

if [ "$MODE" == "file" ]; then
    /inference/functions/chamber_classifier.py --dicom_file /data.local/data/DICOM_DATA/OTHER_DATA/1.2.276.0.76.3.1.53.4903599.2.20180509121635.1345.132179/I59KGNG0 --pipeline_config /data.local/giangh/pipeline/inference/functions/hint_generator_pipeline_config/2C_example.json --out_dir tmp
elif [ "$MODE" == "dir" ]; then
    ./inference/functions/chamber_classifier.py --dicom_dir /data.local/data/DICOM_DATA/OTHER_DATA/1.2.840.113619.2.239.7255.1443093521.0.111/ --pipeline_config /data.local/giangh/pipeline/inference/functions/hint_generator_pipeline_config/2C_example.json --out tmp
fi
