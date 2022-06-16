python segmentation/trainers/train_actor_critic.py \
    --actor_model_class MobileNetv2_DeepLabv3 \
    --critic_model_class CriticMobileNetv2_DeepLabv3 \
    --epochs 40 \
    --trainset /data.local/tuannm/data/2020-02-04_2C/video_20200223_cropped_train_dev \
    --testset /data.local/tuannm/data/2020-02-04_2C/video_20200223_cropped_test \
    --model_prefix tmp/cropped_actor_critic \
    --device "cuda:0" \
    --batch_size 4 \
    --base_lr 1e-3 \
    --n_future_neighbors 6
    --critic_epochs 1