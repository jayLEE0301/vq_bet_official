"""Environments using ur3 robot."""
import logging
from typing import Callable, Dict, Optional, Tuple, Union

import d4rl.kitchen.adept_envs as adept_envs
import einops
import gym
import numpy as np
import torch

import gym_custom
from gym_custom import spaces
from gym_custom.envs.custom.ur_utils import URScriptWrapper_SingleUR3 as URScriptWrapper
from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from collections import OrderedDict
import itertools
from dataset import UR3TrajectoryDataset
from gym.envs.registration import register


def ur3_mask_robots():
    def transform(input):
        assert len(input) == 3, "Input length must be 4: (obs, act, goal)"
        # assume obs, act, mask, goal
        goal = input[-1].clone()
        goal[..., [0, 1]] = 0
        return (*input[:-1], goal)

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
    relay_traj = UR3TrajectoryDataset(data_directory, onehot_goals=True)
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
            return_goal[:, :2] = 0  # mask robot position
            env.set_task_goal(return_goal[-1, 2:6].cpu().numpy())
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


# End effector Constraint
class UprightConstraint(NullObjectiveBase):
    def __init__(self):
        pass

    def _evaluate(self, SO3):
        axis_des = np.array([0, 0, -1])
        axis_curr = SO3[:, 2]
        return 1.0 - np.dot(axis_curr, axis_des)


class UR3Wrapper(gym.Wrapper):
    def __init__(self, env, id, goal_cond=False):
        super(UR3Wrapper, self).__init__(env)
        self.servoj_args, self.speedj_args = (
            {"t": None, "wait": None},
            {"a": 5, "t": None, "wait": None},
        )
        self.PID_gains = {
            "servoj": {"P": 1.0, "I": 0.5, "D": 0.2},
            "speedj": {"P": 0.20, "I": 10.0},
        }
        self.ur3_scale_factor = np.array([5, 5, 5, 5, 5, 5])
        self.gripper_scale_factor = np.array([1.0])
        self.env = URScriptWrapper(
            env, self.PID_gains, self.ur3_scale_factor, self.gripper_scale_factor
        )
        self.max_episode_steps = 1000
        # For action bound
        self.command_limits = {
            "movej": [np.array([-0.04, -0.04, 0]), np.array([0.04, 0.04, 0])]  # [m]
        }
        self.action_space = self._set_action_space()["movej"]
        # Set motor gain scale
        self.env.wrapper_right.ur3_scale_factor[:6] = [
            24.52907494,
            24.02851783,
            25.56517597,
            14.51868608,
            23.78797503,
            21.61325463,
        ]
        self.null_obj_func = UprightConstraint()
        self.state = None
        self.id = id
        self.absolute_pos = True
        self.completion_ids = []
        self.goal_1 = np.array([0.0, -0.25])
        self.goal_2 = np.array([0.0, -0.40])
        self.goal_cond = goal_cond

    def convert_action_to_space(self, action_limits):
        if isinstance(action_limits, dict):
            space = spaces.Dict(
                OrderedDict(
                    [
                        (key, self.convert_action_to_space(value))
                        for key, value in self.command_limits.items()
                    ]
                )
            )
        elif isinstance(action_limits, list):
            low = action_limits[0]
            high = action_limits[1]
            space = gym_custom.spaces.Box(low, high, dtype=action_limits[0].dtype)
        else:
            raise NotImplementedError(type(action_limits), action_limits)
        return space

    def _set_action_space(self):
        return self.convert_action_to_space({"_": self.command_limits})

    def set_task_goal(self, task_goal):
        self.goal_1 = task_goal[:2]
        self.goal_2 = task_goal[2:]

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        self.done = False
        self.goal1_achieved = False
        self.goal2_achieved = False
        self.episode_steps = 0
        self.dt = 1
        self.state = np.array([0.45, -0.325, 0.3, -0.25, 0.3, -0.40])
        print("UR3 block push episode start!")
        self.completion_ids = []
        self.goal_1 = np.array([0.0, -0.25])
        self.goal_2 = np.array([0.0, -0.40])
        return self.state.copy()

    def step(self, action):
        if self.absolute_pos:
            action = np.concatenate([action, [0.8]])
            q_right_des, _, _, _ = self.env.inverse_kinematics_ee(
                action, self.null_obj_func, arm="right"
            )
        else:
            action = action / 25  # neural network action output sacle : -1 ~ 1
            action = np.concatenate((action, np.zeros(1)))
            curr_pos = np.concatenate([self.state[:2], [0.8]])
            q_right_des, _, _, _ = self.env.inverse_kinematics_ee(
                curr_pos + action, self.null_obj_func, arm="right"
            )
        qvel_right = (q_right_des - self.env.get_obs_dict()["right"]["qpos"]) / self.dt

        next_state, _, done, _ = self.env.step(
            {
                "right": {
                    "speedj": {
                        "qd": qvel_right,
                        "a": self.speedj_args["a"],
                        "t": self.speedj_args["t"],
                        "wait": self.speedj_args["wait"],
                    },
                    "move_gripper_force": {"gf": np.array([15.0])},
                }
            }
        )

        self.episode_steps += 1

        # Ignore the "done" signal if it comes from hitting the time horizon. (max timestep 되었다고 done 해서 next Q = 0 되는 것 방지)
        # mask = 1 if self.episode_steps == self.max_episode_steps else float(not done)
        reward = 0
        if not self.goal1_achieved:
            self.goal1_achieved = self.check_goal1_achieved(next_state)
            if self.goal1_achieved:
                reward = 1
                self.completion_ids.append(1)
        if not self.goal2_achieved:
            self.goal2_achieved = self.check_goal2_achieved(next_state)
            if self.goal2_achieved:
                reward = 1
                self.completion_ids.append(2)
        done = (self.episode_steps >= self.max_episode_steps) or (
            self.goal1_achieved and self.goal2_achieved
        )

        self.state = next_state[:6]
        img = self.env.render(mode="rgb_array")
        info = {
            "image": img,
        }
        if self.goal_cond:
            if done:
                reward = self.calc_reward(next_state)
            else:
                reward = 0
        info["all_completions_ids"] = self.completion_ids

        return self.state.copy(), reward, done, info

    def calc_reward(self, next_state):
        block1_dist = np.linalg.norm(self.goal_1 - next_state[2:4], ord=1)
        block2_dist = np.linalg.norm(self.goal_2 - next_state[4:6], ord=1)
        return -(block1_dist + block2_dist)

    def check_goal1_achieved(self, next_state):
        if np.linalg.norm(self.goal_1 - next_state[2:4]) < 0.05:
            print("goal 1 achieved!")
            return True
        else:
            return False

    def check_goal2_achieved(self, next_state):
        if np.linalg.norm(self.goal_2 - next_state[4:6]) < 0.05:
            print("goal 2 achieved!")
            return True
        else:
            return False

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)
