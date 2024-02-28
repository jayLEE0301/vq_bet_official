import beepy
import numpy as np
import os
import pickle
import pkg_resources
import sys
import time
import traceback
import warnings

import gym_custom
from gym_custom.envs.real.ur.interface import URScriptInterface, convert_action_to_space, convert_observation_to_space, COMMAND_LIMITS
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no

TEMP_DISABLE_LEFT_ARM = False

class DualUR3RealEnv(gym_custom.Env):
    
    # class variables
    ur3_nqpos, gripper_nqpos = 6, 1 # per ur3/gripper joint pos dim
    # ur3_nqvel, gripper_nqvel = 6, 1 # per ur3/gripper joint vel dim
    # ur3_nact, gripper_nact = 6, 1 # per ur3/gripper action dim
    ENABLE_COLLISION_CHECKER = False

    def __init__(self, host_ip_right, host_ip_left, rate):
        self.host_ip_right = host_ip_right
        self.host_ip_left = host_ip_left
        self.interface_right = URScriptInterface(host_ip_right, alias='right')
        self.interface_left = URScriptInterface(host_ip_left, alias='left')
        self.rate = ROSRate(rate)
        self.dt = 1/rate
        self.goal_pos = np.array([0, 0, 0, 0, 0, 0])

        self._define_class_variables()
        
        if self.ENABLE_COLLISION_CHECKER:
            mujoco_py_version = pkg_resources.get_distribution('mujoco-py').version
            assert mujoco_py_version == '1.50.1.68', 'mujoco-py version 1.50.1.68 required! got %s instead'%(mujoco_py_version)
            try:
                import mujoco_py
            except:
                print('mujoco_py required for ENABLE_COLLISION_CHECKER!')
                sys.exit()
            self._define_collision_checker_variables()

        self.run_before_rate_sleep() # clear _run_before_rate_sleep_func
        self.run_before_rate_sleep_return = {}

    def _define_class_variables(self):
        '''overridable method'''
        # Initial position/velocity
        self._init_qpos = np.zeros([12])
        self._init_qvel = np.zeros([12])
        self._init_gripperpos = np.zeros([2])
        self._init_grippervel = np.zeros([2])

        # Variables for forward/inverse kinematics
        # https://www.universal-robots.com/articles/ur-articles/parameters-for-calculations-of-kinematics-and-dynamics/
        self.kinematics_params = {}

        # 1. Last frame aligns with (right/left)_ee_link body frame
        # self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819]) # in m
        # 2. Last frame aligns with (right/left)_gripper:hand body frame
        self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819+0.12]) # in m
        self.kinematics_params['a'] = np.array([0, -0.24365, -0.21325, 0, 0, 0]) # in m
        self.kinematics_params['alpha'] =np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]) # in rad
        self.kinematics_params['offset'] = np.array([0, 0, 0, 0, 0, 0])
        self.kinematics_params['ub'] = np.array([2*np.pi for _ in range(6)])
        self.kinematics_params['lb'] = np.array([-2*np.pi for _ in range(6)])
        
        path_to_pkl = os.path.join(os.path.dirname(__file__), 'ur/dual_ur3_kinematics_params.pkl')
        if os.path.isfile(path_to_pkl):
            kinematics_params_from_pkl = pickle.load(open(path_to_pkl, 'rb'))
            self.kinematics_params['T_wb_right'] = kinematics_params_from_pkl['T_wb_right']
            self.kinematics_params['T_wb_left'] = kinematics_params_from_pkl['T_wb_left']
        else:
            raise FileNotFoundError('No such file: %s. Run MuJoCo-based simulated environment to generate file.'%(path_to_pkl))
        
        # Define spaces
        self.action_space = self._set_action_space()
        obs = self._get_obs()
        self.observation_space = self._set_observation_space(obs)

        # Misc
        self._episode_step = None

    def _define_collision_checker_variables(self):
        from gym_custom.envs.custom.dual_ur3_env import DualUR3Env
        self.collision_env = DualUR3Env()
        self.collision_env.reset()

    def _is_collision(self, right_ur3_qpos):
        if self.ENABLE_COLLISION_CHECKER:
            qpos = self.collision_env.sim.data.qpos.copy()
            qvel = np.zeros_like(self.collision_env.sim.data.qvel)
            qpos[:6] = right_ur3_qpos
            self.collision_env.set_state(qpos, qvel)
            return self.collision_env.data.nefc > 0
        else:
            return False

    #
    # Utilities (general)

    def forward_kinematics_DH(self, q, arm):
        assert len(q) == self.ur3_nqpos

        if arm == 'right':
            T_0_i = self.kinematics_params['T_wb_right']
        elif arm == 'left':
            T_0_i = self.kinematics_params['T_wb_left']
        else:
            raise ValueError('Invalid arm type!')
        T = np.zeros([self.ur3_nqpos+1, 4, 4])
        R = np.zeros([self.ur3_nqpos+1, 3, 3])
        p = np.zeros([self.ur3_nqpos+1, 3])
        # Base frame
        T[0,:,:] = T_0_i
        R[0,:,:] = T_0_i[0:3,0:3]
        p[0,:] = T_0_i[0:3,3]

        for i in range(self.ur3_nqpos):
            ct = np.cos(q[i] + self.kinematics_params['offset'][i])
            st = np.sin(q[i] + self.kinematics_params['offset'][i])
            ca = np.cos(self.kinematics_params['alpha'][i])
            sa = np.sin(self.kinematics_params['alpha'][i])

            T_i_iplus1 = np.array([[ct, -st*ca, st*sa, self.kinematics_params['a'][i]*ct],
                                   [st, ct*ca, -ct*sa, self.kinematics_params['a'][i]*st],
                                   [0, sa, ca, self.kinematics_params['d'][i]],
                                   [0, 0, 0, 1]])
            T_0_i = np.matmul(T_0_i, T_i_iplus1)
            # cf. base frame at i=0
            T[i+1, :, :] = T_0_i
            R[i+1, :, :] = T_0_i[0:3,0:3]
            p[i+1, :] = T_0_i[0:3,3]

        return R, p, T

    def forward_kinematics_ee(self, q, arm):
        R, p, T = self.forward_kinematics_DH(q, arm)
        return R[-1,:,:], p[-1,:], T[-1,:,:]

    def _jacobian_DH(self, q, arm):
        assert len(q) == self.ur3_nqpos
        epsilon = 1e-6
        epsilon_inv = 1/epsilon
        _, ps, _ = self.forward_kinematics_DH(q, arm)
        p = ps[-1,:] # unperturbed position

        jac = np.zeros([3, self.ur3_nqpos])
        for i in range(self.ur3_nqpos):
            q_ = q.copy()
            q_[i] = q_[i] + epsilon
            _, ps_, _ = self.forward_kinematics_DH(q_, arm)
            p_ = ps_[-1,:] # perturbed position
            jac[:, i] = (p_ - p)*epsilon_inv

        return jac

    def inverse_kinematics_ee(self, ee_pos, null_obj_func, arm,
            q_init='current', threshold=0.001, threshold_null=0.001, max_iter=10, epsilon=1e-6
        ):

        '''
        inverse kinematics with forward_kinematics_DH() and _jacobian_DH()
        '''
        # Set initial guess
        if arm == 'right':
            if type(q_init).__name__ == 'ndarray': q = q_init.copy()
            elif q_init == 'current': q = self.interface_right.get_joint_positions()
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        elif arm == 'left':
            if type(q_init).__name__ == 'ndarray': q = q_init.copy()
            elif q_init == 'current': q = self.interface_left.get_joint_positions()
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        else:
            raise ValueError('Invalid arm type!')
        
        SO3, x, _ = self.forward_kinematics_ee(q, arm)
        jac = self._jacobian_DH(q, arm)
        delta_x = ee_pos - x
        err = np.linalg.norm(delta_x)
        null_obj_val = null_obj_func.evaluate(SO3)
        iter_taken = 0

        while True:
            if (err < threshold and null_obj_val < threshold_null) or iter_taken >= max_iter: break
            else: iter_taken += 1

            # pseudo-inverse + null-space approach
            jac_dagger = np.linalg.pinv(jac)
            jac_null = np.eye(self.ur3_nqpos) - np.matmul(jac_dagger, jac) # null space of Jacobian
            phi = np.zeros(self.ur3_nqpos) # find phi (null objective derivative)
            for i in range(self.ur3_nqpos):
                q_perturb = q.copy()
                q_perturb[i] += epsilon
                SO3_perturb, _, _ = self.forward_kinematics_ee(q_perturb, arm)
                null_obj_val_perturb = null_obj_func.evaluate(SO3_perturb)
                phi[i] = (null_obj_val_perturb - null_obj_val)/epsilon
            # update
            delta_x = ee_pos - x
            delta_q = np.matmul(jac_dagger, delta_x) - np.matmul(jac_null, phi)
            q += delta_q
            q = np.minimum(self.kinematics_params['ub'], np.maximum(q, self.kinematics_params['lb'])) # clip within theta bounds
            SO3, x, _ = self.forward_kinematics_ee(q, arm)
            jac = self._jacobian_DH(q, arm)
            null_obj_val = null_obj_func.evaluate(SO3)
            # evaluate
            err = np.linalg.norm(delta_x)
        
        # if iter_taken == max_iter:
        #     warnings.warn('Max iteration limit reached! err: %f (threshold: %f), null_obj_err: %f (threshold: %f)'%(err, threshold, null_obj_val, threshold_null),
        #         RuntimeWarning)
            
        return q, iter_taken, err, null_obj_val

    #
    # Utilities (URScriptInterface related)

    def set_initial_joint_pos(self, q=None):
        if q is None: pass
        elif q == 'current':
            self._init_qpos = np.concatenate([self.interface_right.get_joint_positions(), self.interface_left.get_joint_positions()]).ravel()
        else:
            assert q.shape[0] == 12
            self._init_qpos = q
        print('Initial joint position is set to %s'%(q))
    
    def set_initial_joint_vel(self, qd=None):
        if qd is None: pass
        elif qd == 'current':
            self._init_qvel = np.concatenate([self.interface_right.get_joint_speeds(), self.interface_left.get_joint_speeds()]).ravel()
        else:
            assert qd.shape[0] == 12
            self._init_qvel = qd
        print('Initial joint velocity is set to %s'%(qd))

    def set_initial_gripper_pos(self, g=None):
        if g is None: pass
        elif g == 'current':
            self._init_gripperpos = np.concatenate([self.interface_right.get_gripper_position(), self.interface_left.get_gripper_position()]).ravel()
        else:
            assert g.shape[0] == 2
            self._init_gripperpos = g
        print('Initial gripper position is set to %s'%(g))

    def set_initial_gripper_vel(self, gd=None):
        if gd is None: pass
        elif gd == 'current':
            self._init_gripperpos = np.concatenate([self.interface_right.get_gripper_speed(), self.interface_left.get_gripper_speed()]).ravel()
        else:
            assert gd.shape[0] == 2
            self._init_grippervel = gd
        print('Initial gripper velocity is set to %s'%(gd))

    #
    # Overrided GymEnv methods for compatibility with MujocoEnv methods

    def step(self, action, wait=False):
        assert self._episode_step is not None, 'Must reset before step!'
        # TODO: Send commands to both arms simultaneously?
        for command_type, command_val in action['right'].items():
            getattr(self.interface_right, command_type)(**command_val)
        if not TEMP_DISABLE_LEFT_ARM:
            for command_type, command_val in action['left'].items():
                getattr(self.interface_left, command_type)(**command_val)
        self._episode_step += 1

        self.run_before_rate_sleep_return = self._run_before_rate_sleep_func() # run _run_before_rate_sleep_func
        lag_occurred = self.rate.sleep()
        self.run_before_rate_sleep() # clear _run_before_rate_sleep_func

        # self.interface_right.log('rate.sleep()') ## TEMP TIMESTAMPING
        ob = self._get_obs(wait=wait)
        # self.interface_right.log('_get_obs() for s\'') ## TEMP TIMESTAMPING
        reward = 1.0
        done = False
        if lag_occurred:
            warnings.warn('Desired rate of %dHz is not satisfied! (current rate: %dHz)'%(self.rate._freq, 1/(self.rate._actual_cycle_time) ))
        controller_error = lambda stats: np.any([(stat.safety.StoppedDueToSafety) or (not stat.robot.PowerOn) for stat in stats])
        if not TEMP_DISABLE_LEFT_ARM:
            if controller_error([self.interface_right.get_controller_status(), self.interface_left.get_controller_status()]):
                done_info = self._recover_from_controller_error()
                return ob, reward, True, done_info
            else:
                return ob, reward, done, {}
        else:
            if controller_error([self.interface_right.get_controller_status()]):
                done_info = self._recover_from_controller_error()
                return ob, reward, True, done_info
            else:
                return ob, reward, done, {}
    
    def run_before_rate_sleep(self, func=lambda: {}):
        self._run_before_rate_sleep_func = func

    def reset(self):
        # TODO: Send commands to both arms simultaneously?
        self.interface_right.stopj(a=5, wait=True) # prevent protecive stop(invalid setpoints: sudden stop) error
        if not TEMP_DISABLE_LEFT_ARM:
            self.interface_left.stopj(a=5, wait=True) # prevent protecive stop(invalid setpoints: sudden stop) error
        controller_error = lambda stats: np.any([(stat.safety.StoppedDueToSafety) or (not stat.robot.PowerOn) for stat in stats])
        if not TEMP_DISABLE_LEFT_ARM:
            if controller_error([self.interface_right.get_controller_status(), self.interface_left.get_controller_status()]):
                self._recover_from_controller_error()
        else:
            if controller_error([self.interface_right.get_controller_status()]):
                self._recover_from_controller_error()

        self.goal_pos = np.array([0.1+0.3*np.random.rand(), -0.4, 0.9+0.3*np.random.rand(), -0.1-0.3*np.random.rand(), -0.4, 0.9+0.3*np.random.rand()])
        # self.goal_pos = np.array([0.2, -0.4, 1.0, -0.2, -0.4, 1.0])

        ob = self.reset_model()
        self.rate.reset()
        return ob

    def _recover_from_controller_error(self):
        status_right = self.interface_right.get_controller_status()
        status_left = self.interface_left.get_controller_status()
        robot_right =[attr for attr in dir(status_right.robot) if getattr(status_right.robot, attr)==True]
        safety_right = [attr for attr in dir(status_right.safety) if getattr(status_right.safety, attr)==True]
        robot_left = [attr for attr in dir(status_left.robot) if getattr(status_left.robot, attr)==True]
        safety_left = [attr for attr in dir(status_left.safety) if getattr(status_left.safety, attr)==True]
        status_info = {
            'real_env': True,
            'error_flags_right': robot_right + safety_right,
            'error_flags_left': robot_left + safety_left
        }
        warnings.warn('UR3 controller error! %s'%(status_info))
        print('Resetting UR3 controller...')
        for _ in range(2): # sometimes require 2 calls
            right_reset_done = self.interface_right.reset_controller()
            if not TEMP_DISABLE_LEFT_ARM:
                left_reset_done = self.interface_left.reset_controller()
            else:
                left_reset_done = True
        if (not right_reset_done) or (not left_reset_done):
            while (not right_reset_done) or (not left_reset_done):
                status_right = self.interface_right.get_controller_status()
                status_left = self.interface_left.get_controller_status()
                robot_right =[attr for attr in dir(status_right.robot) if getattr(status_right.robot, attr)==True]
                safety_right = [attr for attr in dir(status_right.safety) if getattr(status_right.safety, attr)==True]
                robot_left = [attr for attr in dir(status_left.robot) if getattr(status_left.robot, attr)==True]
                safety_left = [attr for attr in dir(status_left.safety) if getattr(status_left.safety, attr)==True]
                print('Failed to reset UR3 controller. Manual reset is required.')
                print('ERR_FLAGS: \r\n right - %s, %s \r\n left - %s, %s'%(robot_right, safety_right, robot_left, safety_left))
                # beepy.beep('error')
                if prompt_yes_or_no("Press 'Y' after manual reset to proceed. Press 'n' to terminate program.") is False:
                    print('exiting program!')
                    sys.exit()
                right_reset_done = self.interface_right.reset_controller()
                if not TEMP_DISABLE_LEFT_ARM:
                    left_reset_done = self.interface_left.reset_controller()
                else:
                    left_reset_done = True
            print('UR3 controller manual reset ok')
        else:
            print('UR3 controller reset ok')
        return status_info

    def render(self, mode='human'):
        warnings.warn('Real environment. "Render" with your own two eyes!')

    def close(self):
        self.interface_right.close()
        self.interface_left.close()

    def reset_model(self):
        # TODO: Send commands to both arms simultaneously?
        controller_error = lambda stats: np.any([(stat.safety.StoppedDueToSafety) or (not stat.robot.PowerOn) for stat in stats])
        movej_success = False
        while not movej_success:
            try:
                self.interface_right.movej(q=self._init_qpos[:6])
                if not TEMP_DISABLE_LEFT_ARM:
                    self.interface_left.movej(q=self._init_qpos[6:])
                for _ in range(2):
                    obs_dict = self.get_obs_dict()
                    movej_success = np.linalg.norm(obs_dict['right']['qpos'] - self._init_qpos[:6], np.inf) < np.deg2rad(3)
                    if movej_success: break
                    time.sleep(0.1)
                    self.interface_right.movej(q=self._init_qpos[:6])
                    if not TEMP_DISABLE_LEFT_ARM:
                        self.interface_left.movej(q=self._init_qpos[6:])
                if not movej_success:
                    print('movej of reset_model did not register for some reason..')
                    # beepy.beep('error')
                    if prompt_yes_or_no("Press 'Y' to resend movej command. Press 'n' to terminate program.") is False:
                        print('exiting program!')
                        sys.exit()
            except Exception as e:
                print('hardware error during movej of reset_model')
                traceback.print_exc()
                if not TEMP_DISABLE_LEFT_ARM:
                    if controller_error([self.interface_right.get_controller_status(), self.interface_left.get_controller_status()]):
                        self._recover_from_controller_error()
                else:
                    if controller_error([self.interface_right.get_controller_status()]):
                        self._recover_from_controller_error()
                # beepy.beep('error')
                if prompt_yes_or_no("Press 'Y' after untangling robot arms. Press 'n' to terminate program.") is False:
                    print('exiting program!')
                    sys.exit()
            
        self.interface_right.move_gripper(g=self._init_gripperpos[:1])
        if not TEMP_DISABLE_LEFT_ARM:
            self.interface_left.move_gripper(g=self._init_gripperpos[1:])
        self._episode_step = 0
        return self._get_obs()

    def get_obs_dict(self, wait=False):
        _, curr_right_pos, _ = self.forward_kinematics_ee(self.interface_right.get_joint_positions(wait=wait), 'right')
        _, curr_left_pos, _ = self.forward_kinematics_ee(self.interface_left.get_joint_positions(wait=wait), 'left')
        return {'right': {
                'goal_pos' : self.goal_pos[:3],
                'curr_pos' : curr_right_pos,
                "qpos_sine" : np.sin(self.interface_right.get_joint_positions(wait=wait)),
                "qpos_cosine" : np.cos(self.interface_right.get_joint_positions(wait=wait)),
                'qpos': self.interface_right.get_joint_positions(wait=wait),
                'qvel': self.interface_right.get_joint_speeds(wait=wait),
                'gripperpos': self.interface_right.get_gripper_position(),
                'grippervel': self.interface_right.get_gripper_speed()
            },
            'left': {
                'goal_pos' : self.goal_pos[3:],
                'curr_pos' : curr_left_pos,
                "qpos_sine" : np.sin(self.interface_left.get_joint_positions(wait=wait)),
                "qpos_cosine" : np.cos(self.interface_left.get_joint_positions(wait=wait)),
                'qpos': self.interface_left.get_joint_positions(wait=wait),
                'qvel': self.interface_left.get_joint_speeds(wait=wait),
                'gripperpos': self.interface_left.get_gripper_position(),
                'grippervel': self.interface_left.get_gripper_speed()
            }
        }

    def _get_obs(self, wait=False):
        return self._dict_to_nparray(self.get_obs_dict(wait=wait))

    @staticmethod
    def _dict_to_nparray(obs_dict):
        right = obs_dict['right']
        left = obs_dict['left']
        return np.concatenate([right['goal_pos'], left['goal_pos'], 
                               right['curr_pos'], left['curr_pos'], 
                               right['qpos_sine'], right['qpos_cosine'],
                               left['qpos_sine'] , left['qpos_cosine'],
                               right['qpos'], left['qpos'],
                               right['gripperpos'], left['gripperpos'],
                               right['grippervel'], left['grippervel']]).ravel()


    @staticmethod
    def _nparray_to_dict(obs_nparray):
        return {'right': {
                'qpos': obs_nparray[0:6],
                'qvel': obs_nparray[14:20],
                'gripperpos': obs_nparray[6:7],
                'grippervel': obs_nparray[20:21]
            },
            'left': {
                'qpos': obs_nparray[7:13],
                'qvel': obs_nparray[21:27],
                'gripperpos': obs_nparray[13:14],
                'grippervel': obs_nparray[27:28]
            }
        }

    @staticmethod
    def _set_action_space():
        return convert_action_to_space({'right': COMMAND_LIMITS, 'left': COMMAND_LIMITS})

    @staticmethod
    def _set_observation_space(observation):
        return convert_observation_to_space(observation)


## Examples
def servoj_speedj_example(host_ip_right, host_ip_left, rate):
    real_env = DualUR3RealEnv(host_ip_right=host_ip_right, host_ip_left=host_ip_left, rate=rate)
    real_env.set_initial_joint_pos('current')
    real_env.set_initial_gripper_pos('current')
    if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
        %(np.rad2deg(real_env._init_qpos[:6]), np.rad2deg(real_env._init_qpos[6:]))) is False:
        print('exiting program!')
        sys.exit()
    obs = real_env.reset()
    obs_dict = real_env._nparray_to_dict(obs)
    init_qpos_right, init_qpos_left = obs_dict['right']['qpos'], obs_dict['left']['qpos']
    goal_qpos_right, goal_qpos_left = init_qpos_right.copy(), init_qpos_left.copy()
    goal_qpos_right[-1] += np.pi/2*1.5
    goal_qpos_left[-1] += np.pi/2*1.5
    waypoints_qpos_right = np.linspace(init_qpos_right, goal_qpos_right, rate*2, axis=0)
    waypoints_qpos_left = np.linspace(init_qpos_left, goal_qpos_left, rate*2, axis=0)
    waypoints_qvel_right = np.diff(waypoints_qpos_right, axis=0)*real_env.rate._freq
    waypoints_qvel_left = np.diff(waypoints_qpos_left, axis=0)*real_env.rate._freq
    
    # close-open-close gripper
    print('close')
    real_env.step({'right': {'close_gripper': {}}, 'left': {'close_gripper': {}}})
    time.sleep(3.0)
    print('open')
    real_env.step({'right': {'open_gripper': {}}, 'left': {'open_gripper': {}}})
    time.sleep(3.0)
    print('close')
    real_env.step({'right': {'close_gripper': {}}, 'left': {'close_gripper': {}}})
    time.sleep(5.0)

    if prompt_yes_or_no('servoj to \r\n right: %s deg\r\n left: %s deg\r\n?'
        %(np.rad2deg(goal_qpos_right), np.rad2deg(goal_qpos_left))) is False:
        print('exiting program!')
        sys.exit()
    # servoj example
    print('Testing servoj')
    start = time.time()
    for n, (waypoint_right, waypoint_left) in enumerate(zip(waypoints_qpos_right[1:,:], waypoints_qpos_left[1:,:])):
        real_env.step({
            'right': {
                'servoj': {'q': waypoint_right, 't': 2/real_env.rate._freq, 'wait': False},
                # 'open_gripper': {}
            },
            'left': {
                'servoj': {'q': waypoint_left, 't': 2/real_env.rate._freq, 'wait': False},
                # 'open_gripper': {}
            }
        })
        print('action %d sent!'%(n))
    real_env.step({'right': {'stopj': {'a': 5}}, 'left': {'stopj': {'a': 5}}})
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    time.sleep(5)
    curr_obs_dict = real_env._nparray_to_dict(real_env._get_obs())
    curr_qpos_right, curr_qpos_left = curr_obs_dict['right']['qpos'], curr_obs_dict['left']['qpos']
    print('current - goal qpos is \r\n right: %s deg\r\n left: %s deg'
        %(np.rad2deg(curr_qpos_right - goal_qpos_right), np.rad2deg(curr_qpos_left - goal_qpos_left)))
    time.sleep(5)
    print('Moving to initial position...')
    real_env.step({'right': {'movej': {'q': waypoints_qpos_right[0,:]}}, 'left': {'movej': {'q': waypoints_qpos_left[0,:]}}})
    print('done!')

    if prompt_yes_or_no('speedj to \r\n right: %s deg\r\n left: %s deg?\r\n'
        %(np.rad2deg(goal_qpos_right), np.rad2deg(goal_qpos_left))) is False:
        print('exiting program!')
        sys.exit()
    # speedj example
    print('Testing speedj')
    start = time.time()
    for n, (waypoint_right, waypoint_left) in enumerate(zip(waypoints_qvel_right, waypoints_qvel_left)):
        real_env.step({
            'right': {
                'speedj': {'qd': waypoint_right, 'a': 5, 't': 2/real_env.rate._freq, 'wait': False},
                # 'open_gripper': {}
            },
            'left': {
                'speedj': {'qd': waypoint_left, 'a': 5, 't': 2/real_env.rate._freq, 'wait': False},
                # 'open_gripper': {}
            }
        })
        print('action %d sent!'%(n))
    real_env.step({'right': {'stopj': {'a': 5}}, 'left': {'stopj': {'a': 5}}})
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    time.sleep(5)
    curr_obs_dict = real_env._nparray_to_dict(real_env._get_obs())
    curr_qpos_right, curr_qpos_left = curr_obs_dict['right']['qpos'], curr_obs_dict['left']['qpos']
    print('current - goal qpos is \r\n right: %s deg\r\n left: %s deg'
        %(np.rad2deg(curr_qpos_right - goal_qpos_right), np.rad2deg(curr_qpos_left - goal_qpos_left)))
    time.sleep(5)
    print('Moving to initial position...')
    real_env.step({'right': {'movej': {'q': waypoints_qpos_right[0,:]}}, 'left': {'movej': {'q': waypoints_qpos_left[0,:]}}})
    print('done!')
    
    # open-close-open gripper
    print('open')
    real_env.step({'right': {'close_gripper': {}}, 'left': {'close_gripper': {}}})
    time.sleep(3.0)
    print('close')
    real_env.step({'right': {'open_gripper': {}}, 'left': {'open_gripper': {}}})
    time.sleep(3.0)
    print('open')
    real_env.step({'right': {'close_gripper': {}}, 'left': {'close_gripper': {}}})
    time.sleep(5.0)

if __name__ == "__main__":
    servoj_speedj_example(host_ip_right='192.168.5.102', host_ip_left='192.168.5.101', rate=10)
    pass