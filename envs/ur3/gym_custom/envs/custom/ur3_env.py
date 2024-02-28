'''
Environments used for debugging
'''

import numpy as np
import os
import warnings
warnings.simplefilter('always', DeprecationWarning)

from gym_custom import utils
from gym_custom.envs.mujoco import MujocoEnv


class UR3PracticeEnv(MujocoEnv, utils.EzPickle):

    def __init__(self):
        warnings.warn('This environment is for debugging and is only preserved for compatibility with legacy source code!', DeprecationWarning)
        utils.EzPickle.__init__(self)
        xml_filename = 'ur3_pedestalmounted.xml'
        # xml_filename = 'shared_config.xml'
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


class UR3PracticewithGripperEnv(MujocoEnv, utils.EzPickle):

    def __init__(self):
        warnings.warn('This environment is for debugging and is only preserved for compatibility with legacy source code!', DeprecationWarning)
        utils.EzPickle.__init__(self)
        xml_filename = 'ur3_withgripper_pedestalmounted.xml'
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


class UR3dschoEnv(MujocoEnv, utils.EzPickle): # sanity check

    def __init__(self):
        warnings.warn('This environment is for debugging and is only preserved for compatibility with legacy source code!', DeprecationWarning)
        utils.EzPickle.__init__(self)
        xml_filename = 'ur3_pick_and_place_dscho.xml'
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'ur3', xml_filename)
        MujocoEnv.__init__(self, fullpath, 5)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
        
        
