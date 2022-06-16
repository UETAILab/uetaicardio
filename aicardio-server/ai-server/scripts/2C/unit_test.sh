export $(pwd):$(pwd)/PyTorch_YOLOv3

# # test datasets
# python lv_inference/datasets.py --dicom_path "/data.local/tuannm/data/test_2020_03_13/GLS/0001__1.2.840.113663.1500.393838829248544842578148983107292161/IM_0006-2C"\
#                                 --visualize_dir "visualize-unit-test/datasets"
                                
# test pipeline
python lv_inference/lv_pipeline.py --detector_model_weights "/data.local/data/models/YOLO-2C/0.7435_yolov3_ckpt_75.pth" \
            --full_image_module_name "segmentation" \
            --full_image_segmentation_class "AuxUNet" \
            --full_image_segmentation_weights "/data.local/data/models/segmentation/full2C/full_aux_giangunet_invhd_0009_0.8604_best.pth" \
            --cropped_image_module_name "segmentation" \
            --cropped_image_segmentation_class "Resnet101DeeplabV3" \
            --cropped_image_segmentation_weights "/data.local/data/models/segmentation/cropped2C/cropped_aux_hd_resnet101_b8_vloss_0079_0.8707_best.pth" \
            --dicom_path "/data.local/tuannm/data/test_2020_03_13/GLS/0001__1.2.840.113663.1500.393838829248544842578148983107292161/IM_0006-2C" \
            --visualize_dir "visualize-unit-test/pivot_tracker" \
            --csv_output_file "visualize-unit-test/pivot_tracker.csv"
