import copy
import numpy as np
import pickle
import os
import warnings

import gym_custom
from gym_custom import utils
from gym_custom.envs.mujoco import MujocoEnv

import matplotlib.pyplot as plt


# For Simulation environment

class SingleUR3XYEnv(MujocoEnv, utils.EzPickle):

    # class variables
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets/ur3/single_ur3_base.xml')
    mujocoenv_frame_skip = 1
    # state
    ur3_nqpos, gripper_nqpos = 6, 10 # per ur3/gripper joint pos dim
    ur3_nqvel, gripper_nqvel = 6, 10 # per ur3/gripper joint vel dim
    objects_nqpos = [7, 7, 7, 7] # there is 4 objects on the table, each object has qpos = (3trans + 4quat)
    objects_nqvel = [6, 6, 6, 6] # there is 4 objects on the table, each object has qpos = (3trans + 3rota)
    # action
    ur3_nact, gripper_nact = 6, 2 # per ur3/gripper action dim
    ENABLE_COLLISION_CHECKER = False
    # ee position
    curr_pos = np.array([0, 0])
    curr_pos_block = np.array([1,1])


    def __init__(self):
        if self.ENABLE_COLLISION_CHECKER:
            self._define_collision_checker_variables()
        self._ezpickle_init()
        self._mujocoenv_init()
        self._check_model_parameter_dimensions()
        self._define_class_variables()

    def _ezpickle_init(self):
        '''overridable method'''
        utils.EzPickle.__init__(self)

    def _mujocoenv_init(self):
        '''overridable method'''
        MujocoEnv.__init__(self, self.mujoco_xml_full_path, self.mujocoenv_frame_skip)

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert self.ur3_nqpos + self.gripper_nqpos + sum(self.objects_nqpos) == self.model.nq, 'Number of qpos elements mismatch'
        assert self.ur3_nqvel + self.gripper_nqvel + sum(self.objects_nqvel) == self.model.nv, 'Number of qvel elements mismatch'
        assert self.ur3_nact + self.gripper_nact == self.model.nu, 'Number of action elements mismatch'

    def _define_class_variables(self):
        '''overridable method'''
        # Initial position for UR3
        self.init_qpos[0:self.ur3_nqpos] = \
        np.array([ 1.22096933, -1.3951761, 1.4868261, -2.01667739, 0.84679318, -0.00242263])
        # np.array([90, -45, 135, -180, 45, 0])*np.pi/180.0 # right arm
       

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
        
        self.kinematics_params['T_wb_right'] = np.eye(4)
        self.kinematics_params['T_wb_right'][0:3,0:3] = self.sim.data.get_body_xmat('right_arm_rotz').reshape([3,3]).copy()
        self.kinematics_params['T_wb_right'][0:3,3] = self.sim.data.get_body_xpos('right_arm_rotz').copy()

        path_to_pkl = os.path.join(os.path.dirname(__file__), '../real/ur/dual_ur3_kinematics_params.pkl')
        if not os.path.isfile(path_to_pkl):
            pickle.dump(self.kinematics_params, open(path_to_pkl, 'wb'))

    def _define_collision_checker_variables(self):
        self.collision_env = self

    def _is_collision(self, right_ur3_qpos):
        is_collision = False
        if self.ENABLE_COLLISION_CHECKER:
            qpos_original, qvel_original = self.collision_env.sim.data.qpos.copy(), self.collision_env.sim.data.qvel.copy()
            qpos = self.collision_env.sim.data.qpos.copy()
            qvel = np.zeros_like(self.collision_env.sim.data.qvel)
            qpos[:6] = right_ur3_qpos
            self.collision_env.set_state(qpos, qvel)
            is_collision = self.collision_env.sim.data.nefc > 0
            self.collision_env.set_state(qpos_original, qvel_original)
        return is_collision

    #
    # Utilities (general)

    def forward_kinematics_DH(self, q, arm):
        assert len(q) == self.ur3_nqpos
        self._define_class_variables()

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
            elif q_init == 'current': q = self._get_ur3_qpos()[:self.ur3_nqpos]
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        elif arm == 'left':
            if type(q_init).__name__ == 'ndarray': q = q_init.copy()
            elif q_init == 'current': q = self._get_ur3_qpos()[self.ur3_nqpos:]
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
        # print(iter_taken)
        return q, iter_taken, err, null_obj_val

    #
    # Utilities (MujocoEnv related)

    def get_body_se3(self, body_name):
        R = self.sim.data.get_body_xmat(body_name).reshape([3,3]).copy()
        p = self.sim.data.get_body_xpos(body_name).copy()
        T = np.eye(4)
        T[0:3,0:3] = R
        T[0:3,3] = p

        return R, p, T

####
    def _get_ur3_qpos(self):
        return np.concatenate([self.sim.data.qpos[0:self.ur3_nqpos]]).ravel()

    def _get_gripper_qpos(self):
        return np.concatenate([self.sim.data.qpos[self.ur3_nqpos:self.ur3_nqpos+self.gripper_nqpos]]).ravel()

    def _get_ur3_qvel(self):
        return np.concatenate([self.sim.data.qvel[0:self.ur3_nqvel]]).ravel()

    def _get_gripper_qvel(self):
        return np.concatenate([self.sim.data.qvel[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def _get_ur3_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[0:self.ur3_nqvel]]).ravel()

    def _get_gripper_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def _get_ur3_constraint(self):
        return np.concatenate([self.sim.data.qfrc_constraint[0:self.ur3_nqvel]]).ravel()

    def _get_ur3_actuator(self):
        return np.concatenate([self.sim.data.qfrc_actuator[0:self.ur3_nqvel]]).ravel()

    def _get_obs(self):
        '''overridable method'''
        return np.concatenate([self.curr_pos, self.curr_pos_block, np.sin(self._get_ur3_qpos()), np.cos(self._get_ur3_qpos()),
                               self._get_gripper_qpos(), self._get_gripper_qvel()]).ravel()
    
####

    def get_ur3_qpos(self):
        return np.concatenate([self.sim.data.qpos[0:self.ur3_nqpos]]).ravel()

    def get_gripper_qpos(self):
        return np.concatenate([self.sim.data.qpos[self.ur3_nqpos:self.ur3_nqpos+self.gripper_nqpos]]).ravel()

    def get_ur3_qvel(self):
        return np.concatenate([self.sim.data.qvel[0:self.ur3_nqvel]]).ravel()

    def get_gripper_qvel(self):
        return np.concatenate([self.sim.data.qvel[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def get_ur3_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[0:self.ur3_nqvel]]).ravel()

    def get_gripper_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def get_ur3_constraint(self):
        return np.concatenate([self.sim.data.qfrc_constraint[0:self.ur3_nqvel]]).ravel()

    def get_ur3_actuator(self):
        return np.concatenate([self.sim.data.qfrc_actuator[0:self.ur3_nqvel]]).ravel()

    def get_obs(self):
        '''overridable method'''
        return np.concatenate([self.curr_pos, self.curr_pos_block, np.sin(self._get_ur3_qpos()), np.cos(self._get_ur3_qpos()),
                               self._get_gripper_qpos(), self._get_gripper_qvel()]).ravel()

####

    def get_obs_dict(self):
        '''overridable method'''
        _, curr_pos, _ = self.forward_kinematics_ee(self._get_ur3_qpos(), 'right')
        return {'right': {
                'curr_pos_block': self.curr_pos_block,
                'curr_pos': curr_pos,
                "qpos_sine"  : np.sin(self._get_ur3_qpos()),
                "qpos_cosine": np.cos(self._get_ur3_qpos()),
                'qpos': self._get_ur3_qpos(),
                'qvel': self._get_ur3_qvel(),
                'gripperpos': self._get_gripper_qpos(),
                'grippervel': self._get_gripper_qvel()
            }
        }

    # Overrided MujocoEnv methods

    def step(self, a):
        '''overridable method'''

        # cube pos
        id_cube_6 = self.sim.model.geom_name2id("cube_6")
        self.curr_pos_block = np.concatenate([self.sim.data.geom_xpos[id_cube_6][:2]])

        # gripper pos
        SO3, curr_pos, _ = self.forward_kinematics_ee(self._get_ur3_qpos()[:self.ur3_nqpos], 'right')
        self.curr_pos = curr_pos[:2]

        # goal pos
        goal_pos = np.array([0.0, -0.35])

        # reward action
        reward_acion = -0.0000001*np.linalg.norm(a)

        # reward pos & reward reaching
        reward_pos = -np.linalg.norm(self.curr_pos_block - goal_pos)
        reward_reaching = -np.linalg.norm(self.curr_pos_block - self.curr_pos)

        if np.linalg.norm(self.curr_pos_block - goal_pos) < 0.05:
            reward_pos = 100
            reward_reaching = 0
            print("goal in")

        # reward_bound 
        is_inside_bound = self.is_inside_bound(self.curr_pos[0], self.curr_pos[1], -0.1, -0.6, 0.75, 0.60)
        if is_inside_bound == False:
            reward_bound = -1.0
        else:
            reward_bound = 0.0

        reward = reward_acion + reward_pos + 0.01*reward_reaching + reward_bound

        for i in range(12):
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            self.set_state(qpos, qvel)
            self.do_simulation(a, self.frame_skip)

        # depth related    
        depth = self.render(mode = 'depth_array', camera_id = 2)
        plt.imshow(depth, cmap='viridis')  # 'viridis'는 컬러맵 선택입니다.
        plt.colorbar()  # 컬러바를 추가합니다.
        plt.title('Depth Image')  # 그래프 제목을 설정합니다.
        plt.show()

        ob = self._get_obs()
        done = False
       
        return ob, reward, done, {}


    def reset_model(self):
        '''overridable method'''

        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)

        # qpos[-21] =  0.10 +0.35*np.random.rand() # x  0 ~ 0.55
        # qpos[-20] = -0.30 -0.15*np.random.rand() # y -0.5 ~ -0.1

        block_pos_candi = np.array([[0.15, -0.3], [0.3, -0.3], [0.15, -0.4], [0.3, -0.4], [0.225, -0.35]])

        rand_idx = np.random.randint(5)
        qpos[-21:-19] = block_pos_candi[rand_idx]

        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        '''overridable method'''
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def quaternion_to_euler(self, quat):

        quat = np.array(quat)
        q0, q1, q2, q3 = quat
        rotation_matrix = np.array([
            [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
            [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
            [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
        ])
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = np.arcsin(-rotation_matrix[2, 0])
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        
        return np.array(yaw)

    def is_inside_bound(self, x, y, start_x, start_y, bound_width, bound_height):
        return start_x <= x < start_x + bound_width and start_y <= y < start_y + bound_height

def test_video_record(env):
    import time
    stime = time.time()
    env.reset()

    for i in range(10000):
        action = env.action_space.sample()
        env.step(action)
        # print(action) # action dim == 8
        env.render()
        print('step: %d'%(i))
    ftime = time.time()


if __name__ == '__main__':
    env = gym_custom.make('single-ur3-larr-v0')
    test_video_record(env)