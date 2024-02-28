"""Environments using ant robot."""
import logging
from typing import Callable, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch

from dataset import AntTrajectoryDataset
from gym.envs.registration import register


def ant_mask_targets():
    def transform(input):
        assert len(input) == 3, "Input length must be 4: (obs, act, goal)"
        # assume obs, act, mask, goal
        obs = input[0].clone()
        obs[..., 29:37] = 0
        goal = input[-1].clone()
        goal[..., :29] = 0
        return (obs, *input[1:-1], goal)

    return transform


def get_goal_fn(
    data_directory: str,
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
    unconditional: bool = False,
) -> Callable[
    [gym.Env, torch.Tensor, torch.Tensor, torch.Tensor],
    Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
        None,
    ],
]:
    empty_tensor = torch.zeros(0)
    if unconditional:
        return lambda env, state, goal_idx, frame_idx: (empty_tensor, {})
    relay_traj = AntTrajectoryDataset(data_directory, onehot_goals=True)
    train_idx, val_idx = get_split_idx(
        len(relay_traj),
        seed=seed,
        train_fraction=train_fraction or 1.0,
    )
    goal_fn = lambda env, state, goal_idx, frame_idx: None
    if goal_conditional == "future":
        assert (
            goal_seq_len is not None
        ), "goal_seq_len must be provided if goal_conditional is 'future'"

        def future_goal_fn(env, state, goal_idx, frame_idx):  # type: ignore
            obs, _, onehot = relay_traj[train_idx[goal_idx]]  # seq_len x obs_dim
            info = {}
            # if frame_idx == 0:
            #     onehot = einops.reduce(onehot, "T C -> C", "max")
            #     info["onehot_goal"] = onehot
            #     env.set_task_goal(onehot)
            return_goal = obs[-goal_seq_len:].clone()
            return_goal[:, :29] = 0
            env.set_task_goal(return_goal[-1, 29:37].cpu().numpy())
            return return_goal, info

        goal_fn = future_goal_fn

    elif goal_conditional == "onehot":

        def onehot_goal_fn(env, state, goal_idx, frame_idx):
            if frame_idx == 0:
                logging.info(f"goal_idx: {train_idx[goal_idx]}")
            _, _, onehot_goals = relay_traj[train_idx[goal_idx]]  # seq_len x obs_dim
            return onehot_goals[min(frame_idx, len(onehot_goals) - 1)], {}

        goal_fn = onehot_goal_fn

    else:
        raise ValueError(f"goal_conditional {goal_conditional} not recognized")

    return goal_fn


def get_split_idx(l, seed, train_fraction=0.95):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(l, generator=rng).tolist()
    l_train = int(l * train_fraction)
    return idx[:l_train], idx[l_train:]


class AntWrapper(gym.Wrapper):
    def __init__(self, env, id, goal_cond):
        super(AntWrapper, self).__init__(env)
        self.id = id
        self.env = env
        if goal_cond:
            self.env.set_goalcond()

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self.num_achieved = 0
        if self.env.goal_cond:
            one_indices = np.random.choice(4, 2, replace=False)
            self.env.set_achieved(one_indices)
            self.num_achieved = 2
        print("Ant Locomotion episode start!")
        return_obs = np.concatenate((obs["observation"], obs["for_vq_bet"]))
        return_obs[29:37] = 0
        return return_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.num_achieved < np.sum(self.env.achieved):
            reward = 1
            self.num_achieved = np.sum(self.env.achieved)
        else:
            reward = 0
        return_obs = np.concatenate((obs["observation"], obs["for_vq_bet"]))
        return_obs[29:37] = 0
        return return_obs, reward, done, info


register(
    id="AntMazeMultimodal-eval-v0",
    entry_point="envs.antenv.ant_maze_multimodal:AntMazeMultimodalEvalEnv",
    max_episode_steps=1200,
    reward_threshold=0.0,
)
