from collections import OrderedDict
from types import SimpleNamespace
import numpy as np

import gym_custom
from gym_custom import spaces
from gym_custom.envs.real.ur.drivers import URBasic

# dscho add
import warnings
from gym_custom.utils import colorize

COMMAND_LIMITS = {
    'movej': [np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -np.inf]),
        np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, np.inf])], # [rad]
    'speedj': [np.array([-np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
        np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])], # [rad/s]
    'move_gripper': [np.array([0]), np.array([1])] # [0: open, 1: close]
}

def convert_action_to_space(action_limits):
    if isinstance(action_limits, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_action_to_space(value))
            for key, value in COMMAND_LIMITS.items()
        ]))
    elif isinstance(action_limits, list):
        low = action_limits[0]
        high = action_limits[1]
        space = gym_custom.spaces.Box(low, high, dtype=action_limits[0].dtype)
    else:
        raise NotImplementedError(type(action_limits), action_limits)

    return space

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = gym_custom.spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class URScriptInterface(object):
    
    def __init__(self, host_ip, alias=''):
        
        # gripper_kwargs = {
        #     'robot': None,
        #     'payload': 0.85,
        #     'speed': 255, # 0~255
        #     'force': 255,  # 0~255
        #     'socket_host': host_ip,
        #     'socket_name': 'gripper_socket'
        # }

        self.model = URBasic.robotModel.RobotModel()
        # self.comm = URBasic.urScriptExt.UrScriptExt(host=host_ip, robotModel=self.model, **gripper_kwargs)
        # latest dscho modified (2021 0723)
        self.comm = URBasic.urScriptExt.UrScriptExt(host=host_ip, robotModel=self.model)
        self.alias = alias

        logger = URBasic.dataLogging.DataLogging()
        name = logger.AddEventLogging(__name__)
        self.__logger = logger.__dict__[name]
        self.log('init done')

    def __del__(self):
        self.comm.close()

    def close(self):
        self.comm.close()

    ## UR Logger
    def log(self, msg, level='INFO'):
        assert type(msg) == str, 'log message must be string'
        if level == 'INFO': self.__logger.info('[%s] '%(self.alias) + msg)
        elif level == 'DEBUG': self.__logger.debug('[%s] '%(self.alias) + msg)
        elif level == 'ERROR': self.__logger.error('[%s] '%(self.alias) + msg)
        else: pass

    ## UR Controller
    def reset_controller(self):
        return self.comm.reset_error(tsleep=0)

    def get_controller_status(self):
        robot_status = self.comm.robotConnector.RobotModel.RobotStatus()
        safety_status = self.comm.robotConnector.RobotModel.SafetyStatus()
        return SimpleNamespace(robot=robot_status, safety=safety_status)

    ## UR3 manipulator
    def movej(self, q=None, a=1.4, v =1.05, t =0, r =0, wait=True, pose=None):
        '''
        Move to position (linear in joint-space)
        blocking command, not suitable for online control
        '''
        if type(q) == np.ndarray: q = q.tolist()
        self.comm.movej(q=q, a=a, v=v, t=t, r=r, wait=wait, pose=pose)

    def movel(self, *args, **kwargs):
        raise NotImplementedError()

    def movep(self, *args, **kwargs):
        raise NotImplementedError()

    def movec(self, *args, **kwargs):
        raise NotImplementedError()
    
    def servoc(self, *args, **kwargs):
        raise NotImplementedError()

    def servoj(self, q, t =0.008, lookahead_time=0.1, gain=100, wait=True):
        '''
        Servo to position (linear in joint-space)
        non-blocking command, suitable for online control
        '''
        if type(q) == np.ndarray: q = q.tolist()
        self.comm.servoj(q=q, t=t, lookahead_time=lookahead_time, gain=gain, wait=wait)

    def speedj(self, qd, a, t , wait=True):
        '''
        non-blocking command, suitable for online control
        '''
        if type(qd) == np.ndarray: qd = qd.tolist()
        self.comm.speedj(qd=qd, a=a, t=t, wait=wait)

    def speedl(self, *args, **kwargs):
        raise NotImplementedError()
    
    def stopj(self, a, wait=True):
        '''
        '''
        self.comm.stopj(a, wait)

    def stopl(self, *args, **kwargs):
        raise NotImplementedError()

    def get_joint_positions(self, *args, **kwargs):
        # self.log('Reading joint positions...')
        joint_pos = np.array(self.comm.get_actual_joint_positions(*args, **kwargs))
        # self.log('Obtained joint positions')
        return joint_pos

    def get_joint_speeds(self, *args, **kwargs):
        # self.log('Reading joint speeds...')
        joint_speed = np.array(self.comm.get_actual_joint_speeds(*args, **kwargs))
        # self.log('Obtained joint speeds')
        return joint_speed

    ## 2F-85 gripper
    '''
    TODO: dscho
    Gripper commands via urx causes excessive delays under the current implementation.
    Gripper commands should be used in isolation and not alongside UR3 commands for now.
    '''
    def open_gripper(self, *args, **kwargs):
        # self.comm.operate_gripper(0)
        self.move_gripper_position(g=0, *args, **kwargs)

    def close_gripper(self, *args, **kwargs):
        # self.comm.operate_gripper(255)
        self.move_gripper_position(g=255, *args, **kwargs)

    def move_gripper(self, *args, **kwargs):
        '''Compatibility wrapper for move_gripper_position()'''
        return self.move_gripper_position(*args, **kwargs)

    def move_gripper_position(self, g, wait=True):
        self.comm.move_gripper_position(g, wait)

    def move_gripper_velocity(self, gd): # currently, just binary action [open, close]
        # TODO: dscho   
        warnings.warn(colorize('%s: %s'%('WARNING', 'Use move_gripper_position instead of this move_gripper_velocity'), 'yellow'))
        gd = 0
        if gd < 0: # open
            return self.open_gripper()
        elif gd > 0: # close
            return self.close_gripper()
        else: # do nothing
            return None

    def move_gripper_force(self, gf):  # currently, just binary action [open, close]
        # TODO: dscho
        warnings.warn(colorize('%s: %s'%('WARNING', 'Use move_gripper_position instead of this move_gripper_force'), 'yellow'))
        gf = 0
        if gf < 0: # open
            return self.open_gripper()
        elif gf > 0: # close
            return self.close_gripper()
        else: # do nothing
            return None

    def get_gripper_position(self):
        # TODO: dscho
        # return np.array([0.0]) 
        # From the two functions above, I think it's between 0 and 255 : jhpark
        return np.array([self.comm.get_gripper_position()])

    def get_gripper_speed(self):
        # TODO: dscho
        return np.array([0.0])
        # raise NotImplementedError()