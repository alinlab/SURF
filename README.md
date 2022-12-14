# SURF: Semi-supervised Reward Learning with Data Augmentation for Feedback-efficient Preference-based RL (ICLR 2022)
Our implementation is based on the official codebase of PEBBLE (https://github.com/pokaxpoka/B_Pref). If you want to run visual control tasks, please move to the [pixel](https://github.com/alinlab/SURF/tree/pixel) branch.

## Requirements
- Python 3.6
- [MuJoCo](http://mujoco.org/) 2.0

## Install
```
conda env create -f conda_env.yml
conda activate bpref
pip install -e .[docs,tests,extra]
cd custom_dmcontrol
pip install -e .
cd ../custom_dmc2gym
pip install -e .
cd ..
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
pip install pybullet
```

## Run experiments 
## MetaWorld-v2 (Hammer):

### PEBBLE
```
CUDA_VISIBLE_DEVICES=0 python train_PEBBLE.py env=metaworld_hammer-v2 seed=12345 agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=2000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 reward_update=10 num_interact=5000 max_feedback=10000 reward_batch=50 feed_type=1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0
```

### SURF
```
CUDA_VISIBLE_DEVICES=0 python train_PEBBLE_semi_dataaug.py env=metaworld_hammer-v2 seed=12345 agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=2000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 reward_update=20 num_interact=5000 max_feedback=10000 reward_batch=50 feed_type=1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 threshold_u=0.99 mu=4 inv_label_ratio=10
```

## DeepMind Control Suite (Walker walk):

### PEBBLE
```
CUDA_VISIBLE_DEVICES=0 python train_PEBBLE.py  env=walker_walk seed=12345 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 feed_type=1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0
```

### SURF
```
CUDA_VISIBLE_DEVICES=0 python train_PEBBLE_semi_dataaug.py env=walker_walk seed=12345 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=10 inv_label_ratio=100 reward_update=1000 feed_type=1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 threshold_u=0.99 mu=4
```
