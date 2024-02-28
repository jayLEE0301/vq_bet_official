import numpy as np
import os
from gym_custom import utils
from gym_custom.envs.mujoco import MujocoEnv


class PracticeEnv1(MujocoEnv, utils.EzPickle):
    # InvertedPendulum from gym/envs/mujoco

    def __init__(self, actuator_type):
        utils.EzPickle.__init__(self)
        actuator_type_list = ['motor', 'motor(nogravity)', 'position', 'position(dyntype=integrator)', 'velocity', 'nodynamics', 'double']
        if actuator_type == 'motor':
            xml_filename = 'practice1_motor.xml'
        elif actuator_type == 'motor(nogravity)':
            xml_filename = 'practice1_motor_no_gravity.xml'
        elif actuator_type == 'position':
            xml_filename = 'practice1_position.xml'
        elif actuator_type == 'position(dyntype=integrator)':
            xml_filename = 'practice1_position_dyntype_integrator.xml'
        elif actuator_type == 'velocity':
            xml_filename = 'practice1_velocity.xml'
        elif actuator_type == 'nodynamics':
            xml_filename = 'practice1_nodynamics.xml'
        elif actuator_type == 'double':
            xml_filename = 'practice1_double.xml'
        else:
            raise ValueError('actuator_type not in %s'%actuator_type_list)
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'practice', xml_filename)
        MujocoEnv.__init__(self, fullpath, 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        # mjsimstate = self.sim.get_state()
        # return np.concatenate([mjsimstate.qpos, mjsimstate.qvel])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


class PracticeEnv2(MujocoEnv, utils.EzPickle):
    # InvertedDoublePendulumEnv from gym/envs/mujoco

    def __init__(self):
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'practice', 'practice2.xml')
        MujocoEnv.__init__(self, fullpath, 1)
        utils.EzPickle.__init__(self)

    def step(self, action):
        reward = 1.0
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    
class PracticeEnv3(MujocoEnv, utils.EzPickle):
    # Wiper

    def __init__(self):
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'practice', 'practice3.xml')
        MujocoEnv.__init__(self, fullpath, 1)
        utils.EzPickle.__init__(self)

    def step(self, action):
        reward = 1.0
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


class PracticeEnv4(MujocoEnv, utils.EzPickle):

    def __init__(self):
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'practice', 'practice4_kinematics.xml')
        MujocoEnv.__init__(self, fullpath, 1)
        utils.EzPickle.__init__(self)

    def step(self, action):
        reward = 1.0
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)*0.0,
            self.init_qvel + self.np_random.randn(self.model.nv) * .1*0.0
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent