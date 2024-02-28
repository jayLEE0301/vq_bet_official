import numpy as np
import argparse
from collections import deque
from gym import spaces

# import create_maze_env


def get_reward_fn(env_name):  # we don't use this function
    if env_name in ["AntMaze", "AntMazeSmall-v0", "AntMazeComplex-v0", "AntPush"]:
        return lambda obs, goal: -(np.sum(np.square(obs[:2] - goal)) ** 0.5)
    elif env_name == "AntMazeSparse":
        return lambda obs, goal: float(np.sum(np.square(obs[:2] - goal)) ** 0.5 < 1)
    elif env_name == "AntFall":
        return lambda obs, goal: -(np.sum(np.square(obs[:3] - goal)) ** 0.5)
    else:
        assert False, "Unknown env"


def get_success_fn(env_name):  # we don't use this function
    if env_name in ["AntMaze", "AntMazeComplex-v0", "AntPush", "AntFall"]:
        return lambda reward: reward > -5.0
    elif env_name in ["AntMazeSmall-v0"]:
        return lambda reward: reward > -2.5
    elif env_name == "AntMazeSparse":
        return lambda reward: reward > 1e-6
    else:
        assert False, "Unknown env"


class GatherEnv(object):
    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.count = 0

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        obs = self.base_env.reset()
        self.count = 0
        return {
            "observation": obs.copy(),
            "achieved_goal": obs[:2],
            "desired_goal": None,
        }

    def step(self, a):
        obs, reward, done, info = self.base_env.step(a)
        self.count += 1
        next_obs = {
            "observation": obs.copy(),
            "achieved_goal": obs[:2],
            "desired_goal": None,
        }
        return next_obs, reward, done or self.count >= 500, info

    def compute_reward(self, achieved_goal, goal, info=None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = np.array(dist) > self.distance_threshold
            return -rs.astype(np.float32)
        else:
            return -dist

    def goal_distance(self, achieved_goal, goal):
        if achieved_goal.ndim == 1:
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    @property
    def action_space(self):
        return self.base_env.action_space


class EnvWithGoal(object):
    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.success_fn = get_success_fn(env_name)
        self.goal = None
        self.distance_threshold = (
            0.5
            if env_name
            in ["AntMaze", "AntMazeSmall-v0", "AntMazeComplex-v0", "AntPush", "AntFall"]
            else 1
        )
        self.count = 0
        self.early_stop = (
            False
            if env_name
            in ["AntMaze", "AntMazeSmall-v0", "AntMazeComplex-v0", "AntPush", "AntFall"]
            else True
        )
        self.early_stop_flag = False

    def seed(self, seed):
        self.base_env.seed(seed)

    def rand_goal(self):
        if self.env_name == "AntMaze":
            while True:
                self.goal = np.random.uniform(low=-4.0, high=20.0, size=2)
                if not (
                    (self.goal[0] < 12) and (self.goal[1] > 4) and (self.goal[1] < 12)
                ):
                    break
        elif self.env_name == "AntMazeSmall-v0":
            while True:
                self.goal = np.random.uniform(low=-2.0, high=10.0, size=2)
                if not (
                    (self.goal[0] < 6) and (self.goal[1] > 2) and (self.goal[1] < 6)
                ):
                    break
        elif self.env_name == "AntMazeComplex-v0":
            while True:
                self.goal = np.random.uniform(low=[-4, -4], high=[52, 52], size=2)
                x = self.goal[0]
                y = self.goal[1]
                if (
                    ((((-4 < x) and (x < 20)) or ((28 < x) and (x < 52))) and (y < 4))
                    or (
                        (((12 < x) and (x < 20)) or ((28 < x) and (x < 52)))
                        and ((4 < y) and (y < 12))
                    )
                    or (
                        (((-4 < x) and (x < 20)) or ((36 < x) and (x < 44)))
                        and ((12 < y) and (y < 20))
                    )
                    or (
                        (((-4 < x) and (x < 4)) or ((28 < x) and (x < 44)))
                        and ((20 < y) and (y < 28))
                    )
                    or (
                        (
                            ((-4 < x) and (x < 4))
                            or ((12 < x) and (x < 20))
                            or ((28 < x) and (x < 36))
                        )
                        and ((28 < y) and (y < 36))
                    )
                    or (
                        (((-4 < x) and (x < 36)) or ((44 < x) and (x < 52)))
                        and ((36 < y) and (y < 44))
                    )
                    or (
                        (
                            ((-4 < x) and (x < 4))
                            or ((12 < x) and (x < 20))
                            or ((28 < x) and (x < 52))
                        )
                        and ((44 < y) and (y < 52))
                    )
                ):
                    break

        else:
            raise NameError("rand goal error")

    def reset(self):
        self.early_stop_flag = False
        obs = self.base_env.reset()
        self.count = 0
        if self.evaluate:
            if self.env_name == "AntMaze":
                self.goal = np.array([0.0, 16.0])
            elif self.env_name == "AntMazeSmall-v0":
                self.goal = np.array([0.0, 8.0])
            elif self.env_name == "AntMazeComplex-v0":
                goal_seed = np.random.randint(4)
                if goal_seed == 0:
                    self.goal = np.array([40.0, 8.0])
                elif goal_seed == 1:
                    self.goal = np.array([16.0, 48.0])
                elif goal_seed == 2:
                    self.goal = np.array([40.0, 48.0])
                else:
                    self.goal = np.array([16.0, 32.0])
            else:
                raise NameError("rand goal error")
        else:
            self.rand_goal()
        self.desired_goal = self.goal
        return {
            "observation": obs.copy(),
            "achieved_goal": obs[:2],
            "desired_goal": self.desired_goal,
        }

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        reward = self.high_reward_func(obs[:2], self.goal, info)
        if self.early_stop and self.success_fn(reward):
            self.early_stop_flag = True
        self.count += 1
        done = self.early_stop_flag and self.count % 10 == 0
        next_obs = {
            "observation": obs.copy(),
            "achieved_goal": obs[:2],
            "desired_goal": self.desired_goal,
        }
        if self.env_name in ["AntMaze", "AntPush", "AntFall"]:
            info["is_success"] = self.goal_distance(obs[:2], self.desired_goal) <= 5
            done = done or self.count >= 600
        elif self.env_name in ["AntMazeComplex-v0"]:
            info["is_success"] = self.goal_distance(obs[:2], self.desired_goal) <= 5
            done = done or self.count >= 2000
        elif self.env_name == "AntMazeSmall-v0":
            info["is_success"] = self.goal_distance(obs[:2], self.desired_goal) <= 2.5
            done = done or self.count >= 600
        return next_obs, reward, done, info

    def render(self):
        self.base_env.render()

    def compute_reward(
        self, achieved_goal, goal, info=None, sparse=False, threshold=None
    ):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            if threshold is None:
                rs = np.array(dist) > self.distance_threshold
            else:
                rs = np.array(dist) > threshold
            return -rs.astype(np.float32)
        else:
            return -dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def high_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False) * 0.5

    def goal_distance(self, achieved_goal, goal):
        if achieved_goal.ndim == 1:
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

    @property
    def action_space(self):
        return self.base_env.action_space
