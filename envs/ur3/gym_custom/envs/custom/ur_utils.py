import copy
import numpy as np
import warnings
warnings.simplefilter('always', DeprecationWarning)

from gym_custom.core import ActionWrapper


### Base class

class NullObjectiveBase(object):
    '''
    Base class for inverse kinematics null objective

    Must overload __init__() and _evaluate()
    '''

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, SO3):
        return self._evaluate(SO3)

    def _evaluate(self, SO3):
        raise NotImplementedError


class URScriptWrapper(ActionWrapper):
    '''
    Base class for all UR Script action wrappers. Ensures compatiblitiy between simulated and real environments.
    '''

    # class variables
    ur3_command_types = ['servoj', 'speedj']
    gripper_command_types = ['move_gripper_position', 'move_gripper_velocity', 'move_gripper_force']
    command_type_list = ur3_command_types + gripper_command_types
    ur3_torque_limit = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])

    def __init__(self, env, PID_gains, ur3_scale_factor, gripper_scale_factor, _env_getters=None):
        super().__init__(env)

        # Aggregate of all env-related attributes/functions that are referenced in this class
        if _env_getters is None:
            self._env_getters = {
                'qpos': self.env._get_ur3_qpos,
                'qvel': self.env._get_ur3_qvel,
                'qbias': self.env._get_ur3_bias,
                'qconstraint': self.env._get_ur3_constraint,
                'gpos': lambda : np.array([self.env._get_gripper_qpos()[2], self.env._get_gripper_qpos()[7]]),
                'gvel': lambda : np.array([self.env._get_gripper_qvel()[2], self.env._get_gripper_qvel()[7]]),
                'gbias': lambda : np.array([self.env._get_gripper_bias()[2], self.env._get_gripper_bias()[7]]),
                'dt': self.env.dt
            }
        else:
            self._env_getters = _env_getters

        self._initialize_attributes(copy.deepcopy(PID_gains), ur3_scale_factor.copy(), gripper_scale_factor.copy())

    def _initialize_attributes(self, PID_gains, ur3_scale_factor, gripper_scale_factor):
        self.ur3_scale_factor = ur3_scale_factor # ndof
        self.gripper_scale_factor = gripper_scale_factor # ngripper
        self.ndof, self.ngripperdof = ur3_scale_factor.shape[0], gripper_scale_factor.shape[0]
        # assert self.ndof == self.env.ur3_nact and self.ngripperdof == self.env.gripper_nact, 'DOF mismatch' # jgkim_temp
        
        self.servoj_gains = PID_gains.get('servoj', {'P': None, 'I': None, 'D': None})
        self.speedj_gains = PID_gains.get('speedj', {'P': None, 'I': None, 'D': None})
        self.ur3_err_integ, self.gripper_err_integ = np.zeros([self.ndof]), np.zeros([2*self.ngripperdof])
        # self.ur3_err_integ_limits, self.gripper_err_integ_limits = [-2.5, 2.5], [-1.0, 1.0]
        
        self.command_type_ur3 = None
        self.command_type_gripper = None

    def _clear_integration_term(self, ur_command=None, tag='', force=True):
        if tag != '': tag = '_' + tag
        if force:
            self.ur3_err_integ = np.zeros_like(self.ur3_err_integ)
            self.gripper_err_integ = np.zeros_like(self.gripper_err_integ)
        else:
            assert ur_command is not None
            for command_type in ur_command.keys():
                if command_type in self.ur3_command_types:
                    if command_type != getattr(self, 'command_type_ur3' + tag):
                        setattr(self, 'command_type_ur3' + tag, command_type)
                        setattr(self, 'ur3_err_integ' + tag, np.zeros([self.ndof]))
                elif command_type in self.gripper_command_types:
                    if command_type != getattr(self, 'command_type_gripper' + tag):
                        setattr(self, 'command_type_gripper' + tag, command_type)
                        setattr(self, 'gripper_err_integ' + tag, np.zeros([2*self.ngripperdof]))
                else:
                    raise ValueError('Invalid command type!')

    def action(self, ur_command):
        if not all([command_type in self.command_type_list for command_type in ur_command.keys()]):
            raise ValueError('Invalid command type!')
        self._clear_integration_term(ur_command, force=False) # clear integration term if command type has changed
        
        # generate actions
        actions = {}
        for command_type, command_val in ur_command.items():
            action, entity = getattr(self, command_type)(**command_val)
            if entity in actions.keys(): raise ValueError('Multiple commands for a single entity!')
            else: actions[entity] = action
            
        ur3_action = actions.get('ur3', np.zeros([self.ndof]))
        gripper_action = actions.get('gripper', np.zeros([2*self.ngripperdof]))

        return np.concatenate([ur3_action, gripper_action])

    def movej(self, q=None, a=1.4, v =1.05, t =0, r =0, wait=True, pose=None):
        '''No analogy for this UR Script command'''
        raise NotImplementedError()
        return action, 'ur3'

    def movel(self, *args, **kwargs):
        '''No analogy for this UR Script command'''
        raise NotImplementedError()
        return action, 'ur3'

    def movep(self, *args, **kwargs):
        '''No analogy for this UR Script command'''
        raise NotImplementedError()
        return action, 'ur3'

    def movec(self, *args, **kwargs):
        '''No analogy for this UR Script command'''
        raise NotImplementedError()
        return action, 'ur3'

    def servoc(self, *args, **kwargs):
        '''No analogy for this UR Script command'''
        raise NotImplementedError()
        return action, 'ur3'

    # def servoj(self, q, t =0.008, lookahead_time=0.1, gain=100, wait=True): # as defined in UR Script
    def servoj(self, q, t=None, lookahead_time=None, gain=None, wait=None): # we only use q here
        '''
        Servo to position (linear in joint-space)
        non-blocking command, suitable for online control
        '''
        # raise NotImplementedError() # TODO: this needs to be tested and/or debugged
        assert q.shape[0] == self.ndof
        # Calculate error
        current_theta = self._env_getters['qpos']()
        # if ur3_command['relative']: # Relative position
        #     theta_dist = np.mod(ur3_command['desired'] - current_theta, 2*np.pi)
        #     err = theta_dist - 2*np.pi*(theta_dist > np.pi)
        # else: # Absolute position
        #     err = ur3_command['desired'] - current_theta
        err = q - current_theta
        err_dot = -self._env_getters['qvel']()
        self.ur3_err_integ = np.clip(self.ur3_err_integ + err*self._env_getters['dt'], -1, 1)

        # Internal forces
        bias = self._env_getters['qbias']()

        # External forces
        constraint = self._env_getters['qconstraint']()
        constraint = np.clip(constraint, -0.50*self.ur3_torque_limit, 0.50*self.ur3_torque_limit)

        # PID controller
        # control_budget_high = self.ur3_torque_limit - (bias - constraint)
        # control_budget_high = np.maximum(control_budget_high, 0)
        # control_budget_low = -self.ur3_torque_limit - (bias - constraint)
        # control_budget_low = np.minimum(control_budget_low, 0)

        PID_control = self.ur3_scale_factor*(self.servoj_gains['P']*err + self.servoj_gains['I']*self.ur3_err_integ + self.servoj_gains['D']*err_dot)

        # scale_upper = np.min(np.where(PID_control > 0, control_budget_high/PID_control, np.inf))
        # scale_lower = np.min(np.where(PID_control < 0, control_budget_high/PID_control, np.inf))
        # rescale = min(scale_lower, scale_upper, 1)
        rescale = 1

        action = rescale*PID_control + bias - constraint*0
        return action, 'ur3'

    # def speedj(self, qd, a, t , wait=True): # as defined in UR Script
    def speedj(self, qd, a=None, t=None, wait=None): # we only use qd here
        '''
        from URScript API Reference v3.5.4
            qd: joint speeds (rad/s)
            a: joint acceleration [rad/s^2] (of leading axis)
            t: time [s] before the function returns (optional)
        '''
        # raise NotImplementedError() # TODO: this needs to be tested and/or debugged
        assert qd.shape[0] == self.ndof
        # Calculate error
        current_thetadot = self._env_getters['qvel']()
        err = qd - current_thetadot
        self.ur3_err_integ = np.clip(self.ur3_err_integ + err*self._env_getters['dt'], -0.02, 0.02)

        # Internal forces
        bias = self._env_getters['qbias']()

        # External forces
        constraint = self._env_getters['qconstraint']()
        constraint = np.clip(constraint, -0.50*self.ur3_torque_limit, 0.50*self.ur3_torque_limit)

        # PID controller
        # control_budget_high = self.ur3_torque_limit - (bias - constraint)
        # control_budget_high = np.maximum(control_budget_high, 0)
        # control_budget_low = -self.ur3_torque_limit - (bias - constraint)
        # control_budget_low = np.minimum(control_budget_low, 0)

        PI_control = self.ur3_scale_factor*(self.speedj_gains['P']*err + self.speedj_gains['I']*self.ur3_err_integ)

        # scale_upper = np.min(np.where(PID_control > 0, control_budget_high/PID_control, np.inf))
        # scale_lower = np.min(np.where(PID_control < 0, control_budget_high/PID_control, np.inf))
        # rescale = min(scale_lower, scale_upper, 1)
        rescale = 1

        action = rescale*PI_control + bias - constraint*0
        return action, 'ur3'

    def speedl(self, *args, **kwargs):
        '''No analogy for this UR Script command'''
        raise NotImplementedError()
        return action, 'ur3'

    def stopj(self, a, wait=True):
        '''No analogy for this UR Script command'''
        return self.speedj(qd=np.zeros[self.ndof], a=a, wait=wait)

    def stopl(self, *args, **kwargs):
        '''No analogy for this UR Script command'''
        raise NotImplementedError()
        return action, 'ur3'

    def open_gripper(self, *args, **kwargs):
        raise NotImplementedError()
        self.move_gripper_position()
        return action, 'gripper'

    def close_gripper(self, *args, **kwargs):
        raise NotImplementedError()
        self.move_gripper_position()
        return action, 'gripper'

    def move_gripper_position(self, g):
        raise NotImplementedError() # TODO: this needs to be tested and/or debugged
        assert g.shape[0] == self.ngripperdof
        err = np.concatenate([g, g]) - self._env_getters['gpos']()
        action = self.gripper_scale_factor*err + self._env_getters['gbias']() # P control
        return action, 'gripper'
    
    def move_gripper_velocity(self, gd):
        raise NotImplementedError() # TODO: this needs to be tested and/or debugged
        assert gd.shape[0] == self.ngripperdof
        err = np.concatenate([gd, gd]) - self._env_getters['gvel']()
        action = self.gripper_scale_factor*err + self._env_getters['gbias']() # P control
        return action, 'gripper'

    def move_gripper_force(self, gf):
        # raise NotImplementedError() # TODO: this needs to be tested and/or debugged
        assert gf.shape[0] == self.ngripperdof
        action = np.concatenate([gf, gf]) + self._env_getters['gbias']()
        return action, 'gripper'

    def reset(self, **kwargs):
        self._clear_integration_term()
        return self.env.reset(**kwargs)


### Derived class

class URScriptWrapper_DualUR3(ActionWrapper):

    def __init__(self, env, PID_gains, ur3_scale_factor, gripper_scale_factor):
        super().__init__(env)

        # cf. f = lambda : np.zeros([5])
        wrapper_right_env_getters = {
            'qpos': lambda : env.get_ur3_qpos()[:env.ur3_nqpos],
            'qvel': lambda : env.get_ur3_qvel()[:env.ur3_nqvel],
            'qbias': lambda : env.get_ur3_bias()[:env.ur3_nqvel],
            'qconstraint': lambda : env.get_ur3_constraint()[:env.ur3_nqvel],
            'gpos': lambda : np.array([self.env.get_gripper_qpos()[2], self.env.get_gripper_qpos()[7]]),
            'gvel': lambda : np.array([self.env.get_gripper_qvel()[2], self.env.get_gripper_qvel()[7]]),
            'gbias': lambda : np.array([self.env.get_gripper_bias()[2], self.env.get_gripper_bias()[7]]),
            'dt': env.dt
        }
        wrapper_left_env_getters = {
            'qpos': lambda : env.get_ur3_qpos()[env.ur3_nqpos:],
            'qvel': lambda : env.get_ur3_qvel()[env.ur3_nqvel:],
            'qbias': lambda : env.get_ur3_bias()[env.ur3_nqvel:],
            'qconstraint': lambda : env.get_ur3_constraint()[env.ur3_nqvel:],
            'gpos': lambda : np.array([self.env.get_gripper_qpos()[12], self.env.get_gripper_qpos()[17]]),
            'gvel': lambda : np.array([self.env.get_gripper_qvel()[12], self.env.get_gripper_qvel()[17]]),
            'gbias': lambda : np.array([self.env.get_gripper_bias()[12], self.env.get_gripper_bias()[17]]),
            'dt': env.dt
        }
        self.wrapper_right = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor, _env_getters=wrapper_right_env_getters)
        self.wrapper_left = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor, _env_getters=wrapper_left_env_getters)
        
        # disable step() and reset() for wrappers
        self.wrapper_right.env = None
        self.wrapper_left.env = None

    def action(self, ur_command): # note that ur_command is dictionary
        right_action = self.wrapper_right.action(ur_command['right'])
        left_action = self.wrapper_left.action(ur_command['left'])
        right_ur3_action, right_gripper_action = right_action[:self.wrapper_right.ndof], right_action[self.wrapper_right.ndof:] # ndof, 2*ngripperdof
        left_ur3_action, left_gripper_action = left_action[:self.wrapper_left.ndof], left_action[self.wrapper_left.ndof:] # ndof, 2*ngripperdof

        return np.concatenate([right_ur3_action, left_ur3_action, right_gripper_action, left_gripper_action])

    def reset(self, **kwargs):
        self.wrapper_right._clear_integration_term()
        self.wrapper_left._clear_integration_term()
        return self.env.reset(**kwargs)


class URScriptWrapper_SingleUR3(ActionWrapper):

    def __init__(self, env, PID_gains, ur3_scale_factor, gripper_scale_factor):
        super().__init__(env)

        # cf. f = lambda : np.zeros([5])
        wrapper_right_env_getters = {
            'qpos': lambda : env.get_ur3_qpos()[:env.ur3_nqpos],
            'qvel': lambda : env.get_ur3_qvel()[:env.ur3_nqvel],
            'qbias': lambda : env.get_ur3_bias()[:env.ur3_nqvel],
            'qconstraint': lambda : env.get_ur3_constraint()[:env.ur3_nqvel],
            'gpos': lambda : np.array([self.env.get_gripper_qpos()[2], self.env.get_gripper_qpos()[7]]),
            'gvel': lambda : np.array([self.env.get_gripper_qvel()[2], self.env.get_gripper_qvel()[7]]),
            'gbias': lambda : np.array([self.env.get_gripper_bias()[2], self.env.get_gripper_bias()[7]]),
            'dt': env.dt
        }

        self.wrapper_right = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor, _env_getters=wrapper_right_env_getters)

        # disable step() and reset() for wrappers
        self.wrapper_right.env = None

    def action(self, ur_command): # note that ur_command is dictionary
        right_action = self.wrapper_right.action(ur_command['right'])
        right_ur3_action, right_gripper_action = right_action[:self.wrapper_right.ndof], right_action[self.wrapper_right.ndof:] # ndof, 2*ngripperdof

        return np.concatenate([right_ur3_action, right_gripper_action])

    def reset(self, **kwargs):
        self.wrapper_right._clear_integration_term()
        return self.env.reset(**kwargs)

class URScriptWrapper_SingleUR3_LEFT(ActionWrapper):

    def __init__(self, env, PID_gains, ur3_scale_factor, gripper_scale_factor):
        super().__init__(env)

        # cf. f = lambda : np.zeros([5])
        wrapper_left_env_getters = {
            'qpos': lambda : env.get_ur3_qpos()[:env.ur3_nqpos],
            'qvel': lambda : env.get_ur3_qvel()[:env.ur3_nqvel],
            'qbias': lambda : env.get_ur3_bias()[:env.ur3_nqvel],
            'qconstraint': lambda : env.get_ur3_constraint()[:env.ur3_nqvel],
            'gpos': lambda : np.array([self.env.get_gripper_qpos()[2], self.env.get_gripper_qpos()[7]]),
            'gvel': lambda : np.array([self.env.get_gripper_qvel()[2], self.env.get_gripper_qvel()[7]]),
            'gbias': lambda : np.array([self.env.get_gripper_bias()[2], self.env.get_gripper_bias()[7]]),
            'dt': env.dt
        }

        self.wrapper_left = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor, _env_getters=wrapper_left_env_getters)

        # disable step() and reset() for wrappers
        self.wrapper_left.env = None

    def action(self, ur_command): # note that ur_command is dictionary
        left_action = self.wrapper_left.action(ur_command['left'])
        left_ur3_action, left_gripper_action = left_action[:self.wrapper_left.ndof], left_action[self.wrapper_left.ndof:] # ndof, 2*ngripperdof

        return np.concatenate([left_ur3_action, left_gripper_action])

    def reset(self, **kwargs):
        self.wrapper_left._clear_integration_term()
        return self.env.reset(**kwargs)

# class URScriptWrapper_DualUR3_fail(URScriptWrapper):
#     '''
#     UR Script action wrapper for DualUR3Env
#     '''

#     # class variables
#     ur3_command_types = ['servoj', 'speedj']
#     gripper_command_types = ['move_gripper_position', 'move_gripper_velocity', 'move_gripper_force']
#     command_type_list = ur3_command_types + gripper_command_types
#     ur3_torque_limit = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0, 50.0, 50.0, 25.0, 10.0, 10.0, 10.0])

#     def _initialize_attributes(self, PID_gains, ur3_scale_factor, gripper_scale_factor):
#         self.ur3_scale_factor = np.concatenate([ur3_scale_factor, ur3_scale_factor]) # 2 x ndof
#         self.gripper_scale_factor = np.concatenate([gripper_scale_factor, gripper_scale_factor]) # 2 x ngripper
#         self.ndof, self.ngripperdof = ur3_scale_factor.shape[0], gripper_scale_factor.shape[0]
#         assert self.ndof == self.env.ur3_nact and self.ngripperdof == self.env.gripper_nact, 'DOF mismatch'
        
#         self.servoj_gains = PID_gains.get('servoj', {'P': None, 'I': None, 'D': None})
#         self.speedj_gains = PID_gains.get('speedj', {'P': None, 'I': None, 'D': None})
#         self.ur3_err_integ, self.gripper_err_integ = np.zeros([2*self.ndof]), np.zeros([2*self.ngripperdof])
#         # self.ur3_err_integ_limits, self.gripper_err_integ_limits = [-2.5, 2.5], [-1.0, 1.0]
#         self.ur3_err_integ_right, self.ur3_err_integ_left = self.ur3_err_integ[:self.ndof], self.ur3_err_integ[self.ndof:]
#         self.gripper_err_integ_right, self.gripper_err_integ_left = self.gripper_err_integ[:self.ngripperdof], self.gripper_err_integ[self.ngripperdof:]
        
#         self.command_type_ur3_right, self.command_type_ur3_left = None, None
#         self.command_type_gripper_right, self.command_type_gripper_left = None, None

#     def action(self, ur_command):
#         if not all([command_type in self.command_type_list for command_type in ur_command['right'].keys()]):
#             raise ValueError('Invalid command type!')
#         if not all([command_type in self.command_type_list for command_type in ur_command['left'].keys()]):
#             raise ValueError('Invalid command type!')
#         self._clear_integration_term(ur_command['right'], tag='right', force=False) # clear integration term if command type has changed
#         self._clear_integration_term(ur_command['left'], tag='left', force=False) # clear integration term if command type has changed

#         # generate actions
#         actions = {}
#         for command_type, command_val in ur_command['right'].items():
#             key, action = getattr(self, command_type)(**command_val)
#             key = key + '_right'
#             if key in actions.keys(): raise ValueError('Multiple commands for a single entity!')
#             else: actions[key] = action
#         for command_type, command_val in ur_command['left'].items():
#             key, action = getattr(self, command_type)(**command_val)
#             key = key + '_left'
#             if key in actions.keys(): raise ValueError('Multiple commands for a single entity!')
#             else: actions[key] = action
#         ur3_right_action = actions.get('ur3_right', np.zeros([self.ndof]))
#         ur3_left_action = actions.get('ur3_left', np.zeros([self.ndof]))
#         gripper_right_action = actions.get('gripper_right', np.zeros([self.ngripperdof]))
#         gripper_left_action = actions.get('gripper_left', np.zeros([self.ngripperdof]))

#         return np.concatenate([ur3_right_action, ur3_left_action, gripper_right_action, gripper_left_action])


# ### Deprecated class

# class URScriptWrapper_DualUR3_deprecated(ActionWrapper):
#     '''
#     UR Script Wrapper for DualUR3Env:
#         Original action wrapper for DualUR3Env (command type fixed for both arms and command type fixed for both grippers)
#     '''
#     def __init__(self, env, PID_gains, ur3_scale_factor, gripper_scale_factor):
#         warnings.warn('This wrapper has been deprecated and is only preserved for compatibility with legacy source code!', DeprecationWarning)
#         super().__init__(env)
#         self.ur3_scale_factor = np.concatenate([ur3_scale_factor, ur3_scale_factor])
#         self.gripper_scale_factor = np.concatenate([gripper_scale_factor, gripper_scale_factor])
#         self.ndof, self.ngripperdof = ur3_scale_factor.shape[0], gripper_scale_factor.shape[0]
#         assert self.ndof == self.env.ur3_nact and self.ngripperdof == self.env.gripper_nact, 'DOF mismatch'
        
#         self.PID_gains = copy.deepcopy(PID_gains)
#         self.ur3_err_integ, self.gripper_err_integ = 0.0, 0.0
#         # self.ur3_err_integ_limits, self.gripper_err_integ_limits = [-2.5, 2.5], [-1.0, 1.0]
        
#         self.ur3_command_type, self.gripper_command_type = None, None

#         self.ur3_torque_limit = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0, 50.0, 50.0, 25.0, 10.0, 10.0, 10.0])

#     def action(self, ur_command, relative=False):
#         ur3_command_type_list = ['servoj', 'speedj']
#         gripper_command_type_list = ['positiong', 'velocityg', 'forceg']

#         if self.ur3_command_type != ur_command['ur3']['type']:
#             self.ur3_err_integ = 0.0 # Clear integration term after command type change
#             self.ur3_command_type = ur_command['ur3']['type']
#         if self.gripper_command_type != ur_command['gripper']['type']:
#             self.gripper_err_integ = 0.0
#             self.gripper_command_type = ur_command['gripper']['type']

#         # UR3 commands
#         if ur_command['ur3']['type'] == ur3_command_type_list[0]:
#             ur3_action = self._servoj(q=ur_command['ur3']['command'], a=None, v=None)
#         elif ur_command['ur3']['type'] == ur3_command_type_list[1]:
#             ur3_action = self._speedj(qd=ur_command['ur3']['command'], a=None)            
#         else:
#             raise ValueError('Invalid UR3 command type!')
#         # gripper commands
#         if ur_command['gripper']['type'] == gripper_command_type_list[0]:
#             gripper_action = self._positiong(q=ur_command['gripper']['command'])
#         elif ur_command['gripper']['type'] == gripper_command_type_list[1]:
#             gripper_action = self._velocityg(qd=ur_command['gripper']['command'])
#         elif ur_command['gripper']['type'] == gripper_command_type_list[2]:
#             gripper_action = self._forceg(qf=ur_command['gripper']['command'])
#         else:
#             raise ValueError('Invalid gripper command type!')
        
#         return np.concatenate([ur3_action, gripper_action])

#     def _servoj(self, q, a, v, t=0.008, lookahead_time=0.1, gain=300):
#         '''
#         from URScript API Reference v3.5.4

#             q: joint positions (rad)
#             a: NOT used in current version
#             v: NOT used in current version
#             t: time where the command is controlling the robot. The function is blocking for time t [S]
#             lookahead_time: time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time
#             gain: proportional gain for following target position, range [100,2000]
#         '''
#         assert q.shape[0] == 2*self.ndof
#         # Calculate error
#         current_theta = self.env._get_ur3_qpos()
#         # if ur3_command['relative']: # Relative position
#         #     theta_dist = np.mod(ur3_command['desired'] - current_theta, 2*np.pi)
#         #     err = theta_dist - 2*np.pi*(theta_dist > np.pi)
#         # else: # Absolute position
#         #     err = ur3_command['desired'] - current_theta
#         err = q - current_theta
#         err_dot = -self.env._get_ur3_qvel()
#         self.ur3_err_integ = np.clip(self.ur3_err_integ + err*self.env.dt, -1, 1)

#         # Internal forces
#         bias = self.env._get_ur3_bias()

#         # External forces
#         constraint = self.env._get_ur3_constraint()
#         constraint = np.clip(constraint, -0.50*self.ur3_torque_limit, 0.50*self.ur3_torque_limit)

#         # PID controller
#         # control_budget_high = self.ur3_torque_limit - (bias - constraint)
#         # control_budget_high = np.maximum(control_budget_high, 0)
#         # control_budget_low = -self.ur3_torque_limit - (bias - constraint)
#         # control_budget_low = np.minimum(control_budget_low, 0)

#         PID_control = self.ur3_scale_factor*(self.PID_gains['P']*err + self.PID_gains['I']*self.ur3_err_integ + self.PID_gains['D']*err_dot)

#         # scale_upper = np.min(np.where(PID_control > 0, control_budget_high/PID_control, np.inf))
#         # scale_lower = np.min(np.where(PID_control < 0, control_budget_high/PID_control, np.inf))
#         # rescale = min(scale_lower, scale_upper, 1)
#         rescale = 1

#         action = rescale*PID_control + bias - constraint
#         return action

#     def _speedj(self, qd, a, t=None):
#         '''
#         from URScript API Reference v3.5.4
#             qd: joint speeds (rad/s)
#             a: joint acceleration [rad/s^2] (of leading axis)
#             t: time [s] before the function returns (optional)
#         '''
#         assert qd.shape[0] == 2*self.ndof
#         # Calculate error
#         current_thetadot = self.env._get_ur3_qvel()
#         err = qd - current_thetadot
#         self.ur3_err_integ = np.clip(self.ur3_err_integ + err*self.env.dt, -0.02, 0.02)

#         # Internal forces
#         bias = self.env._get_ur3_bias()

#         # External forces
#         constraint = self.env._get_ur3_constraint()
#         constraint = np.clip(constraint, -0.50*self.ur3_torque_limit, 0.50*self.ur3_torque_limit)

#         # PID controller
#         # control_budget_high = self.ur3_torque_limit - (bias - constraint)
#         # control_budget_high = np.maximum(control_budget_high, 0)
#         # control_budget_low = -self.ur3_torque_limit - (bias - constraint)
#         # control_budget_low = np.minimum(control_budget_low, 0)

#         PI_control = self.ur3_scale_factor*(self.PID_gains['P']*err + self.PID_gains['I']*self.ur3_err_integ)

#         # scale_upper = np.min(np.where(PID_control > 0, control_budget_high/PID_control, np.inf))
#         # scale_lower = np.min(np.where(PID_control < 0, control_budget_high/PID_control, np.inf))
#         # rescale = min(scale_lower, scale_upper, 1)
#         rescale = 1

#         action = rescale*PI_control + bias - constraint
#         return action

#     def _positiong(self, q):
#         assert q.shape[0] == self.ngripperdof
#         bias = self.env._get_gripper_bias() # Internal forces
#         err = np.array([q[0], q[0], q[1], q[1]]) - self.env._get_gripper_qpos()
#         action = self.gripper_scale_factor*err + np.array([bias[2], bias[7], bias[12], bias[17]]) # P control
#         return action
    
#     def _velocityg(self, qd):
#         assert qd.shape[0] == self.ngripperdof
#         bias = self.env._get_gripper_bias() # Internal forces
#         err = np.array([qd[0], qd[0], qd[1], qd[1]]) - self.env._get_gripper_qvel()
#         action = self.gripper_scale_factor*err + np.array([bias[2], bias[7], bias[12], bias[17]]) # P control
#         return action

#     def _forceg(self, qf):
#         assert qf.shape[0] == self.ngripperdof
#         bias = self.env._get_gripper_bias() # Internal forces
#         action = np.array([qf[0], qf[0], qf[1], qf[1]]) + np.array([bias[2], bias[7], bias[12], bias[17]])
#         return action

#     def reset(self, **kwargs):
#         self.err_integ = 0.0
#         return self.env.reset(**kwargs)