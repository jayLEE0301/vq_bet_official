# VQ-BeT: Behavior Generation with Latent Actions

Official implementation of VQ-BeT: Behavior Generation with Latent Actions.


project website: https://sjlee.cc/vq-bet

<img src="https://github.com/jayLEE0301/vq_bet/assets/30570922/da0654cf-a15a-4ea3-9f90-c389f06e8796">

## Installation

- Make a conda environemnt (We tested on python 3.7 and 3.9) and activate the environment

  ```bash
  conda create -n vq-bet python=3.9
  conda activate vq-bet
  ```

- Clone this repo
  ```bash
  git clone https://github.com/jayLEE0301/vq_bet_official.git
  export PROJ_ROOT=$(pwd)
  ```

- Install [pytorch](https://pytorch.org/get-started/locally/) (We tested on PyTorch 1.12.1 and 2.1.0)

- Install VQ-BeT
  ```bash
  cd $PROJ_ROOT/vq_bet_official
  pip install -r requirements.txt
  pip install -e .
  ```
  or, you can use `sh install.sh`, instead of `pip install -r requirements.txt`

- Install [MuJoCo](https://github.com/deepmind/mujoco) and D4RL

  D4RL can be installed by cloning the repository as follows:
  ```bash
  cd $PROJ_ROOT
  git clone https://github.com/Farama-Foundation/d4rl.git
  cd $PROJ_ROOT/d4rl
  pip install -e .
  cd $PROJ_ROOT/vq_bet_official
  ```

  Also, to run UR3 env, you should install UR3 env
  ```bash
  cd $PROJ_ROOT/vq_bet_official/envs/ur3
  pip install -e .
  cd $PROJ_ROOT/vq_bet_official
  ```

- To enable logging, log in with a wandb account:
  ```bash
  wandb login
  ```
  Alternatively, to disable logging altogether, set the environment variable `WANDB_MODE:`
    ```bash
    export WANDB_MODE=disabled
    ```
## Usage


### Step 0: Download dataset and set dataset path / saving path

- Download datasets [here](https://drive.google.com/file/d/1aHb4kV0mpMvuuApBpVGYjAPs6MCNVTNb/view?usp=sharing).
  - Optionally, use [`gdown`](https://github.com/wkentaro/gdown) to do that: `gdown --fuzzy https://drive.google.com/file/d/1aHb4kV0mpMvuuApBpVGYjAPs6MCNVTNb/view?usp=sharing`.

- Add path to your dataset directory and save path in `./examples/configs/env_vars/env_vars.yaml`.
  ```bash
  # TODO fill these out
  dataset_path: PATH_TO_YOUR_[env_name]_DATASET
  save_path: YOUR_SAVE_PATH
  wandb_entity: YOUR_WANDB_ENTITY
  ```

### Step 1: pretrain vq-vae

- To pretrain Residual VQ, set `config_name="pretrain_[env name]"` in `./examples/pretrain_vqvae.py` and run `pretrain_vqvae.py`. _(e.g., for Goal-cond / Non-goal-cond Kitchen env, `config_name="pretrain_kitchen"`)_

  ```bash
  python examples/pretrain_vqvae.py
  ```
### Step 2: train and evaluate vq-bet

- Add path to your pre-trained Residual VQ in `./examples/configs/train_[env name].yaml` to load them.

  ```bash
  vqvae_load_dir: YOUR_PATH_TO_PRETRAINED_VQVAE/trained_vqvae.pt
  ```


- Then, set `config_name="train_[env name]"` in `./examples/train.py` and run `train.py`  _(e.g., for Non-goal-cond Kitchen env, `config_name="train_kitchen_nongoalcond"`)_
  ```bash
  python examples/train.py
  ```


#### Training visual observation envs:


In this repo, we provide pre-processed embedding vectors with `ResNet18` for the `PushT` and `Kitchen` environments. To train VQ-BeT with visual observation, set `visual_input: true` in `./examples/train_[env name].yaml`. Please not that using freezed embedding could show lower performance compared to fine-tuning `ResNet18` while it is much faster (We will release additional modules for fine-tuning ResNet with VQ-BeT soon).

### (Optional) quick start: evaluating VQ-BeT with pretrained weights (on goal-cond Kitchen env)

If you want to quickly see the performance of VQ-BeT on goal-cond Kitchen env without training it from scratch, please check the description below.

- Download pretrained Residual VQ, and VQ-BeT [here](https://drive.google.com/file/d/1iGRyxwPHMsSVDFGojTiPteU3NVNNXMfP/view?usp=sharing).
  - Optionally, use [`gdown`](https://github.com/wkentaro/gdown) to do that: `gdown --fuzzy https://drive.google.com/file/d/1iGRyxwPHMsSVDFGojTiPteU3NVNNXMfP/view?usp=sharing`.

- Add path to your pre-trained weights in `./examples/configs/train_kitchen_goalcond.yaml` to load them.

  ```bash
  vqvae_load_dir: YOUR_PATH_TO_DOWNLOADED_WEIGHTS/rvq/trained_vqvae.pt
  load_path: YOUR_PATH_TO_DOWNLOADED_WEIGHTS/vq-bet
  ```

- Then, set `config_name="train_kitchen_goalcond"` in `./examples/train.py` and run `train.py`.
  ```bash
  python examples/train.py
  ```

## How can I train VQ-BeT using my own Env? 

NOTE:  You should make your own `./examples/configs/train_[env name].yaml` and `./examples/configs/pretrain_[env name].yaml`

- First, copy `train_your_env.yaml` and `pretrain_your_env.yaml` files from `./examples/configs/template` to `./examples/configs`

- Then, add path to your dataset directory and save path in `./examples/configs/env_vars/env_vars.yaml`. 
  ```bash
  env_vars:
    # TODO fill these out
    dataset_path: PATH_TO_YOUR_[env_name]_DATASET
    save_path: YOUR_SAVE_PATH
    wandb_entity: YOUR_WANDB_ENTITY
  ```
- Also, add the following line under "datasets:" in `./examples/configs/env_vars/env_vars.yaml` containing your environment name. 
  ```bash
  [env_name]: ${env_vars.dataset_path}/[env_name]
  ```

- Then, add your own env file at `examples/[env name]_env.py`. Please note that it should follow [OpenAI Gym](https://github.com/openai/gym) style, and contain `def get_goal_fn` if you are training a goal-conditioned tasks.

- Finally, follow `Step1: pretrain vq-vae` and `Step2: train and evaluate vq-bet` in section `Usage` to pretrain Residual VQ, and train VQ-BeT.

### Tips for hyperparameter tuning on you own env.

During Residual VQ pretraining, the hyperparameters to be determined (in order of importance, with the most important at the top):

1. `action_window_size`:

    - 1 (single-step prediction): Generally sufficient for most environments.

    - 3~5 (multi-step prediction): Can be helpful in environments where action correlation, such as in PushT, is important.

2. `encoder_loss_multiplier`: Adjust this value when the action scale is not between -1 and 1. For example, if the action scale is -100 to 100, a value of 0.01 could be used. If action data is normalized, the default value can be used without adjustment.

3. `vqvae_n_embed`: (10~16 or more) This represents the total possible number of modes, calculated as `vqvae_n_embed^vqvae_groups`. VQ-BeT has robust performance to the size of the dictionary if it is enough to capture the major modes in the dataset (it depends on the tasks, but usually >= 10). Please refer to <em>Section B.1.</em> in the manuscript to see the performance of VQ-BeT with various size of Residual VQ dictionary.


Hyperparameters to be determined during the VQ-BeT training (in order of importance, with the most important at the top):

1. `window_size`: 10 ~ 100: While 10 is suitable in most cases, consider increasing it if a longer observation history is deemed beneficial.

2. `offset_loss_multiplier`: If the action scale is around -1 to 1, the most common value of `offset_loss_multiplier` is 100 (default). Adjust this value if the action scale is not between -1 and 1. For example, if the action scale is -100 to 100, a value of 1 could be used.

3. `secondary_code_multiplier`: The default value is 0.5. Experimenting with values between 0.5 and 3 is recommended. A larger value emphasizes predictions for the secondary code more than offset predictions.


## Common errors and solutions

- Cython compile error
  ```bash
  Cython.Compiler.Errors.CompileError
  ```
  Try `pip install "cython<3"` (https://github.com/openai/mujoco-py/issues/773)

- MuJoCo gcc error
  ```bash
  fatal error: GL/glew.h: No such file or directory
  distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1
  ```

  Try the following solution

  ```bash
  conda install -c conda-forge glew
  conda install -c conda-forge mesalib
  conda install -c menpo glfw3
  ```

  Then add your conda environment include to CPATH (put this in your .bashrc to make it permanent):
  ```bash
  export CPATH=$CONDA_PREFIX/include

  ```
  Finally, install patchelf with `pip install patchelf`

- MuJoCo missing error:
  ```bash
  Error: You appear to be missing MuJoCo.  We expected to find the file here: /home/usr_name/.mujoco/mujoco210 .
  ```
  
  Can be solved by following instructions [here](https://github.com/openai/mujoco-py).

- gladLoadGL error
  ```bash
  Error in call to target 'gym.envs.registration.make':
  FatalError('gladLoadGL error')
  ```
  Try putting `MUJOCO_GL=egl` in front of your command
  
  ```bash
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python examples/train.py 
  ```
---

Our code sourced and modified from [miniBeT implementation](https://github.com/notmahi/miniBET) for [conditional](https://play-to-policy.github.io) and [unconditional behavior transformer](https://mahis.life/bet) Algorithm. Also, we utilizes residual VQ-VAE codes from [Vector Quantization - Pytorch repo](https://github.com/lucidrains/vector-quantize-pytorch), PushT env from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), Ant env base from [DHRL](https://github.com/jayLEE0301/dhrl_official) and UR3 env from [here](https://github.com/snu-larr/dual-ur3-env).