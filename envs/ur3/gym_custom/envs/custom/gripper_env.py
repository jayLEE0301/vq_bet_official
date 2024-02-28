import numpy as np
import os
from gym_custom import utils
from gym_custom.envs.mujoco import MujocoEnv


class Robotiq85PracticeEnv(MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        xml_filename = 'flying_gripper.xml'
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'ur3', xml_filename)
        MujocoEnv.__init__(self, fullpath, 5)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        # qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        # qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent