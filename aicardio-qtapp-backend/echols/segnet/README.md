## Pull repository
```
git clone https://github.com/tqlong/pipeline
cd pipeline/
```

## Train
```
# ~/pipeline
export PYTHONPATH=$(pwd)
python segmentation/trainers/train_single_frame.py --module_name segmentation --model_class MobileNetv2_DeepLabv3 --epochs 10 --trainset /data.local/tuannm/data/2020-02-04_2C/train_dev --testset /data.local/tuannm/data/2020-02-04_2C/test
```
