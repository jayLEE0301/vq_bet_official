import gym

# import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from gym import spaces
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env
import os


class AntMazeMultimodalEvalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    xml_filename = "ant_maze_multimodal.xml"
    goal = np.random.uniform(low=-4.0, high=20.0, size=2)
    mujoco_xml_full_path = os.path.join(
        os.path.dirname(__file__), "assets", xml_filename
    )
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = "sparse"
    distance_threshold = 0.5
    action_threshold = np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0])
    achieved = np.array([0, 0, 0, 0])
    init_xy = np.array([0, 0])
    goal_arr = np.random.uniform(low=-4.0, high=20.0, size=(4, 2))
    max_step = 1200
    completion_ids = []

    def __init__(
        self,
        file_path=None,
        expose_all_qpos=True,
        expose_body_coms=None,
        expose_body_comvels=None,
        seed=0,
    ):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 1200
        self.nb_step = 0
        self.goal_cond = False

        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        """overridable method"""
        assert 15 == self.model.nq, "Number of qpos elements mismatch"
        assert 14 == self.model.nv, "Number of qvel elements mismatch"
        assert 8 == self.model.nu, "Number of action elements mismatch"

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= "1.50":
            return self.sim
        else:
            return self.model

    def set_goalcond(self):
        self.goal_cond = True

    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        done = False
        ob = self._get_obs()
        reward = 0
        for goal_idx in range(4):
            goal = self.goal_arr[goal_idx]
            if self.goal_cond:
                rollout_thr = 1.5
            else:
                rollout_thr = 1.5
            if self.goal_distance(ob["achieved_goal"], goal) <= rollout_thr:
                if self.achieved[goal_idx] == 0:
                    self.achieved[goal_idx] = 1
                    print("ACHIEVED {}th Goal!".format(goal_idx))
                    self.completion_ids.append(goal_idx)
                    ob["for_vq_bet"] = deepcopy(
                        np.concatenate((self.goal_arr.flatten(), self.achieved))
                    )
        self.nb_step = 1 + self.nb_step
        done = bool((self.nb_step > self.max_step) or np.sum(self.achieved) == 4)
        render_img = self.render(
            mode="rgb_array", camera_name="topveiw", width=1024, height=1024
        )
        info = {
            "image": render_img,
            "all_completions_ids": self.completion_ids,
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info=None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = np.array(dist) > self.distance_threshold
            return -rs.astype(np.float32)
        else:
            return -dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.data.qpos.flat[:15],
                self.data.qvel.flat[:14],
            ]
        )
        achieved_goal = obs[:2]
        return {
            "observation": obs.copy(),
            "achieved_goal": deepcopy(achieved_goal),
            "desired_goal": obs.copy(),
            "for_vq_bet": deepcopy(
                np.concatenate((self.goal_arr.flatten(), self.achieved))
            ),
        }

    def set_task_goal(self, goal):
        if self.goal_cond:
            self.goal_arr = np.reshape(goal, (4, 2))

    def set_achieved(self, one_indices):
        if self.goal_cond:
            self.achieved[one_indices] = 1

    def reset_model(self):
        self.goal_arr = np.random.uniform(low=-4.0, high=20.0, size=(4, 2))
        self.goal_arr[0] = np.array(
            [0.0, 0.0]
        )  # + np.random.uniform(low=-1., high=1., size=(2))
        self.goal_arr[1] = np.array(
            [16.0, 0.0]
        )  # + np.random.uniform(low=-1., high=1., size=(2))
        self.goal_arr[2] = np.array(
            [0.0, 16.0]
        )  # + np.random.uniform(low=-1., high=1., size=(2))
        self.goal_arr[3] = np.array(
            [16.0, 16.0]
        )  # + np.random.uniform(low=-1., high=1., size=(2))
        self.achieved = np.array([0, 0, 0, 0])
        # self.set_goal("goal_point")
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * 0.1
        self.init_xy = np.random.uniform(low=7.0, high=9.0, size=2)
        self.init_qpos[:2] = self.init_xy
        qpos[:2] = self.init_xy

        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.0
        self.set_state(qpos, qvel)
        self.nb_step = 0
        self.completion_ids = []

        return self._get_obs()

    def goal_distance(self, achieved_goal, goal):
        if achieved_goal.ndim == 1:
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist
