# SURF: Semi-supervised Reward Learning with Data Augmentation for Feedback-efficient Preference-based RL (ICLR 2022)
This branch is for running visual control tasks of the paper. Our implementation is based on the official codebase of PEBBLE (https://github.com/pokaxpoka/B_Pref) and DrQ-v2 (https://github.com/facebookresearch/drqv2).

## Requirements
- Python 3.8
- [MuJoCo](http://mujoco.org/) 2.0

## Install
```
conda env create -f conda_env_pixel.yml
conda activate drqv2
conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# for using DM Control Suite instead of MetaWorld
pip uninstall -y metaworld numpy
pip install numpy==1.19.2
```

## Run experiments on DeepMind Control Suite:
### Walker walk
```
CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 python pixel_train_PEBBLE_semi.py reward_stack=True reward_lr=3e-5 time_shift=2 time_crop=2 task=walker_walk seed=1 num_train_frames=1000000 num_unsup_frames=0 num_interact=30000 max_feedback=200 reward_batch=10 reward_update=20 inv_label_ratio=5 threshold_u=0.99 lambda_u=0.1
```

### Quadruped walk
```
CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 python pixel_train_PEBBLE_semi.py reward_stack=True reward_lr=3e-5 time_shift=2 time_crop=2 task=quadruped_walk seed=1 num_train_frames=1000000 num_unsup_frames=0 num_interact=30000 max_feedback=1000 reward_batch=50 reward_update=20 inv_label_ratio=5 threshold_u=0.99 lambda_u=0.1
```

### Cheetah run
```
CUDA_VISIBLE_DEVICES=1 EGL_DEVICE_ID=1 python pixel_train_PEBBLE_semi.py reward_stack=True reward_lr=3e-5 time_shift=5 time_crop=5 task=cheetah_run seed=1 num_train_frames=1000000 num_unsup_frames=0 num_interact=20000 max_feedback=1000 reward_batch=25 reward_update=20 inv_label_ratio=5 threshold_u=0.99
```
