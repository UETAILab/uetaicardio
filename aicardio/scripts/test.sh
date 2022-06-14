#!/bin/bash

export PYTHONPATH='.'
export DEBUG=1

echo "-------------------------------------------"
echo "Test classifier"
python echols/classifier.py

echo "Test detector"
python echols/detector.py

echo "Test segmentor"
python echols/segmentor.py

echo "Test main"
python echols/main.py --dicom_path assets/IM_0331

echo "-------------------------------------------"
