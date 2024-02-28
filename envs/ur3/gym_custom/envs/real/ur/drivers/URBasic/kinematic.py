'''
Python 3.x library to control an UR robot through its TCP/IP interfaces
Copyright (C) 2017  Martin Huus Bjerge, Rope Robotics ApS, Denmark

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL "Rope Robotics ApS" BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of "Rope Robotics ApS" shall not be used 
in advertising or otherwise to promote the sale, use or other dealings in this Software 
without prior written authorization from "Rope Robotics ApS".
'''
__author__ = "Martin Huus Bjerge"
__copyright__ = "Copyright 2017, Rope Robotics ApS, Denmark"
__license__ = "MIT License"

import ikpy  as ik
import numpy as np
import sympy as sp
import math
import scipy
from scipy import linalg
from ..URBasic.manipulation import *

pi = np.pi
# Disable the logging stream from ikpy
ik.logs.manager.removeHandler(ik.logs.stream_handler)

def Forwardkin_manip(joints,rob='ur10'):    
    '''
    This function solves forward kinematics, it returns pose vector
    '''
    M,Slist = Robot_parameter_screw_axes(rob)
    thetalist = joints
    fk = FKinFixed(M, Slist, thetalist)
    return np.round(Tran_Mat2Pose(Tran_Mat=fk),4)
    
def Invkine_manip(target_pos,init_joint_pos=[0,0,0, 0,0,0],rob='ur10',tcpOffset=[0,0,0, 0,0,0]):
    '''
    A numerical inverse kinematics routine based on Newton-Raphson method.
    Takes a list of fixed screw axes (Slist) expressed in end-effector body frame, the end-effector zero
    configuration (M), the desired end-effector configuration (T_sd), an initial guess of joint angles
    (thetalist_init), and small positive scalar thresholds (wthresh, vthresh) controlling how close the
    final solution thetas must be to the desired thetas.
    '''
    M,Slist = Robot_parameter_screw_axes(rob)
    wthresh =0.001
    vthresh = 0.0001
    #T_sd = Pose2Tran_Mat(pose=target_pos)
    T_target = Pose2Tran_Mat(pose=target_pos)
    T_tcp = Pose2Tran_Mat(pose=tcpOffset)
    T_sd = np.matmul(T_target, np.linalg.inv(T_tcp))
    thetalist_init = init_joint_pos
    
    ik_init = np.round(IKinFixed(Slist, M, T_sd, thetalist_init, wthresh, vthresh), 3)
    ik = ik_init[-1][:] #choose the closest solution
    
    error = []
    for kk in ik_init:
        out=[]
        for ii in range(6):
            k = np.round((kk[ii]-init_joint_pos[ii])/2/np.pi)
            out.append(kk[ii]-k*2*np.pi)
        error.append(np.abs(np.sum(init_joint_pos-out)))
        
    ix = error==np.min(error)
    ix = np.where(ix==True)[0][0]
    ik = ik_init[ix][:] #choose the closest solution

    out=[]
    for ii in range(6):
        k = np.round((ik[ii]-init_joint_pos[ii])/2/np.pi)
        out.append(ik[ii]-k*2*np.pi)

    
    #error = (pi+ik-init_joint_pos)%(2*pi)-pi
    
    
    print('***************************************************************************')
    print('Pose  : [{: 06.3f}, {: 06.3f}, {: 06.3f},   {: 06.3f}, {: 06.3f}, {: 06.3f}]'.format(*target_pos))
    print('Init q: [{: 06.3f}, {: 06.3f}, {: 06.3f},   {: 06.3f}, {: 06.3f}, {: 06.3f}]'.format(*init_joint_pos))
    print('---------------------------------------------------------------------------')
    print(ik_init)
    print('---------------------------------------------------------------------------')
#    print(error)
#    print('---------------------------------------------------------------------------')
    print('Joints Candidate: [{: 06.3f}, {: 06.3f}, {: 06.3f},   {: 06.3f}, {: 06.3f}, {: 06.3f}]'.format(*ik))
    print('Joints Returned: [{: 06.3f}, {: 06.3f}, {: 06.3f},   {: 06.3f}, {: 06.3f}, {: 06.3f}]'.format(*out))   #init_joint_pos+error))
    print('***************************************************************************')
    
    return out
    #return init_joint_pos+error
    



def rotate_tcp(gradient_vec=[0,0,0]):
    '''
    Convert gradient vector to axis angle vector
    
    Parameters:
    gradient_vec (3D float): [dx, dy, dz] displacement in x, y, z 
    
    Return value:
    axis vector (3D float): [rx, ry, rz]
    
    Exampel:
    Rotate 5deg around x axis and 10 deg around y axis 
    dz = 0.2
    dy = np.tan(10/180*np.pi)*dz
    dx = np.tan(5/180*np.pi)*dz 
    
    r = rotate_tcp(gradient_vec=[dx,dy,dz])
    '''
    
    v_0 = [0,0,1] # this is the UR zero degree angle in the base frame
    v_in_u = gradient_vec / np.linalg.norm(gradient_vec)

    v1_u = v_in_u / np.linalg.norm(v_in_u)
    v2_u = v_0 / np.linalg.norm(v_0)
    rot_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    min_angle  =0.01
    
    if rot_angle < min_angle:
        rotation_vec = [0,0,0]
    elif rot_angle > np.pi - min_angle :
        rotation_vec = [0,-np.pi,0]
    else:
        vec_norm = np.cross(v_0, v_in_u)
        vec_norm = vec_norm/np.linalg.norm(vec_norm)
        rotation_vec = vec_norm * rot_angle
       
    return rotation_vec

    
def Robot_parameter_screw_axes(rob='ur10'):
    '''
    This function defines robot with fixed screw axes(used in manipulation.py)
    rob='ur5'  : ur5
    rob='ur10' : ur10
    https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/actual-center-of-mass-for-robot-17264/ 
    '''
    if str(rob).lower()=='ur5':
        M_ur5 = [[1,0,0,-.81725],[0,0,-1,-.19145],[0,1,0,-.0055],[0,0,0,1]]
        S1_ur5 = [0,0,1,0,0,0]
        S2_ur5 = [0,-1,0,.089159,0,0]
        S3_ur5 = [0,-1,0,.089159,0,.425]
        S4_ur5 = [0,-1,0,.089159,0,.81725]
        S5_ur5 = [0,0,-1,.10915,-.81725,0]  
        S6_ur5 = [0,-1,0,-.0055,0,.81725]
        Slist_ur5 = [S1_ur5,S2_ur5,S3_ur5,S4_ur5,S5_ur5,S6_ur5]
        return M_ur5,Slist_ur5
    elif str(rob).lower()=='ur10':
        M_ur10 = [[1,0,0,-1.1843],[0,0,-1,-0.2561],[0,1,0,0.0116],[0,0,0,1]]
        S1_ur10 = [0,0,1,0,0,0]
        S2_ur10 = [0,-1,0,.1273,0,0]
        S3_ur10 = [0,-1,0,.1273,0,.612]
        S4_ur10 = [0,-1,0,.1273,0,1.1843]
        S5_ur10 = [0,0,-1,.16394,-1.1843,0]  
        S6_ur10 = [0,-1,0,0.01165,0,1.1843]
        Slist_ur10 = [S1_ur10,S2_ur10,S3_ur10,S4_ur10,S5_ur10,S6_ur10]
        return M_ur10,Slist_ur10
    else:
        print('Wrong robot selected')
        return False
    
def Robot_DH_Numerical(rob='ur10',joint=[0,0,0,0,0,0]):
    '''
    This function returns the DH parameter of a robot
    rob='ur5'  : ur5
    rob='ur10' : ur10 
    joint: the robot joint vectors
    '''
    j = joint
    if str(rob).lower()=='ur5':
        dh_ur=np.matrix([
                        [0, pi/2, 0.089159,j[0]],
                        [-0.425,0,0,j[1]],
                        [-0.39225,0,0,j[2]],
                        [0, pi/2, 0.10915,j[3]],
                        [0, -pi/2, 0.09465,j[4]],
                        [0, 0, 0.0823,j[5]]])
        return dh_ur
    elif str(rob).lower()=='ur10':
        dh_ur=np.matrix([
                        [0, pi/2, 0.1273,j[0]],
                        [-0.612 ,0,0,j[1]],
                        [-0.5723 ,0,0,j[2]],
                        [0, pi/2, 0.163941 ,j[3]],
                        [0, -pi/2, 0.1157 ,j[4]],
                        [0, 0, 0.0922,j[5]]])
        return dh_ur
    else:
        print('Wrong robot selected')
        return         
    
    
def Robot_DH_Symbol(rob='ur10' ):
    '''
    This function returns the DH parameter of a robot
    rob='ur5'  : ur5
    rob='ur10' : ur10 
    '''
    # set up our joint angle symbols (6th angle doesn't affect any kinematics)
    q = [sp.Symbol('q%i'%ii) for ii in range(6)]
    if str(rob).lower()=='ur5': 
        dh_ur=np.matrix([
                        [0, pi/2, 0.089159,q[0]],
                        [-0.425,0,0,q[1]],
                        [-0.39225,0,0,q[2]],
                        [0, pi/2, 0.10915,q[3]],
                        [0, -pi/2, 0.09465,q[4]],
                        [0, 0, 0.0823,q[5]]])
        return dh_ur
    elif str(rob).lower()=='ur10':
        dh_ur=np.matrix([
                        [0, pi/2, 0.1273,q[0]],
                        [-0.612 ,0,0,q[1]],
                        [-0.5723 ,0,0,q[2]],
                        [0, pi/2, 0.163941 ,q[3]],
                        [0, -pi/2, 0.1157 ,q[4]],
                        [0, 0, 0.0922,q[5]]])
        return dh_ur
    else:
        print('Wrong robot selected')
        return         

def TransMatrix_DH_Symbol(rob='ur10' ,joint_num=6):
    '''
    This function gives the transfer matrix of robot(DH method)
    rob='ur5'  : ur5
    rob='ur10' : ur10 
    joint_num: the transform matrix for joint_num (from 1 to 6) 
    '''
    T=[]
    # set up our joint angle symbols (6th angle doesn't affect any kinematics)
    q = [sp.Symbol('q%i'%ii) for ii in range(joint_num)]
    dh_ur = Robot_DH_Symbol(rob)
    for ii in range(joint_num):
        #descriptions terms in dh_ur
        #dh_ur[:,0] = a or r
        #dh_ur[:,1] = alpha
        #dh_ur[:,2] = d
        #dh_ur[:,3] = theta (q)
        T.append(sp.Matrix([[sp.cos(dh_ur[ii,3]), -sp.sin(dh_ur[ii,3])*sp.cos(dh_ur[ii,1]),sp.sin(dh_ur[ii,3])*sp.sin(dh_ur[ii,1]),sp.cos(dh_ur[ii,3])*dh_ur[ii,0]],
                            [sp.sin(dh_ur[ii,3]),sp.cos(dh_ur[ii,3])*sp.cos(dh_ur[ii,1]),-sp.cos(dh_ur[ii,3])*sp.sin(dh_ur[ii,1]),sp.sin(dh_ur[ii,3])*dh_ur[ii,0]],
                            [0,sp.sin(dh_ur[ii,1]),sp.cos(dh_ur[ii,1]),dh_ur[ii,2]],
                            [0,0,0,1]
                            ]))
    if joint_num==1:
        return T[0]
    elif joint_num==2:
        return T[0]*T[1]
    elif joint_num==3:
        return T[0]*T[1]*T[2]
    elif joint_num==4:
        return T[0]*T[1]*T[2]*T[3]
    elif joint_num==5:
        return T[0]*T[1]*T[2]*T[3]*T[4]
    elif joint_num==6:
        return T[0]*T[1]*T[2]*T[3]*T[4]*T[5]
    else:
        print('Wrong joint number input')
        return
    


def TransMatrix_DH_Numerical(rob='ur10' ,joint=[0,0,0,0,0,0]):
    '''
    This function gives the transfer matrix of robot(DH method)
    rob='ur5'  : ur5
    rob='ur10' : ur10 
    joint: the robot joint vectors
    '''
    T=[]
    dh_ur = Robot_DH_Numerical(rob, joint)
    for ii in range(6):
        #descriptions terms in dh_ur
        #dh_ur[:,0] = a or r
        #dh_ur[:,1] = alpha
        #dh_ur[:,2] = d
        #dh_ur[:,3] = theta
        T.append(np.matrix([[np.cos(dh_ur[ii,3]), -np.sin(dh_ur[ii,3])*np.cos(dh_ur[ii,1]),np.sin(dh_ur[ii,3])*np.sin(dh_ur[ii,1]),np.cos(dh_ur[ii,3])*dh_ur[ii,0]],
                            [np.sin(dh_ur[ii,3]),np.cos(dh_ur[ii,3])*np.cos(dh_ur[ii,1]),-np.cos(dh_ur[ii,3])*np.sin(dh_ur[ii,1]),np.sin(dh_ur[ii,3])*dh_ur[ii,0]],
                            [0,np.sin(dh_ur[ii,1]),np.cos(dh_ur[ii,1]),dh_ur[ii,2]],
                            [0,0,0,1]
                            ]))
    return np.round(T[0]*T[1]*T[2]*T[3]*T[4]*T[5],4)



def Jacobian_Symbol(rob='ur10' ,joint_num=6):
    '''
    This function returns a 6*6 symbolic jacobian matrix 
    Tx: transfermation matrix
    '''
    # set up our joint angle symbols (6th angle doesn't affect any kinematics) 
    q = [sp.Symbol('q%i'%ii) for ii in range(joint_num)] 

    Tx = TransMatrix_DH_Symbol(rob, joint_num)
    J = []
    # calculate derivative of (x,y,z) wrt to each joint, linear velocity
    for ii in range(joint_num):
        J.append([])
        J[ii].append(sp.simplify(Tx[0,3].diff(q[ii]))) # dx/dq[ii]
        J[ii].append(sp.simplify(Tx[1,3].diff(q[ii]))) # dy/dq[ii]
        J[ii].append(sp.simplify(Tx[2,3].diff(q[ii]))) # dz/dq[ii]
    # orientation part of the Jacobian (compensating for orientations)
    J_orientation = [
                    [0, 0, 1], # joint 0 rotates around z axis
                    [1, 0, 0], # joint 1 rotates around x axis
                    [1, 0, 0], # joint 2 rotates around x axis
                    [1, 0, 0], # joint 3 rotates around x axis
                    [0, 0, 1], # joint 4 rotates around z axis
                    [1, 0, 0]] # joint 5 rotates around x axis
    # add on the orientation information up to the last joint
    for ii in range(joint_num):
        J[ii] = J[ii] + J_orientation[ii]
    return J        


def Jacobian_Numerical(rob='ur10' ,joint=[0,0,0,0,0,0]):
    '''
    This function returns the numerical result of Jacobian
    joint: joint vector
    J is from Jacobian_Symbol
    '''
    q0=joint[0]
    q1=joint[1]
    q2=joint[2]
    q3=joint[3]
    q4=joint[4]
    q5=joint[5]
    if str(rob).lower()=='ur5':
        J = np.matrix([[0.0823*np.sin(q0)*np.sin(q4)*np.cos(q1 + q2 + q3)  - 0.09465*np.sin(q0)*np.sin(q1 + q2 + q3) + 0.425*np.sin(q0)*np.cos(q1) + 0.39225*np.sin(q0)*np.cos(q1 + q2)  + 0.0823*np.cos(q0)*np.cos(q4) + 0.10915*np.cos(q0),  0.0823*np.sin(q0)*np.cos(q4) + 0.10915*np.sin(q0) - 0.0823*np.sin(q4)*np.cos(q0)*np.cos(q1 + q2 + q3)  + 0.09465*np.sin(q1 + q2 + q3)*np.cos(q0) - 0.425*np.cos(q0)*np.cos(q1) - 0.39225*np.cos(q0)*np.cos(q1 + q2), 0, 0, 0, 1], 
                       [ 0.425*np.sin(q1)*np.cos(q0) + 0.0823*np.sin(q4)*np.sin(q1 + q2 + q3)*np.cos(q0) + 0.39225*np.sin(q1 + q2)*np.cos(q0)  + 0.09465*np.cos(q0)*np.cos(q1 + q2 + q3), 0.425*np.sin(q0)*np.sin(q1) + 0.0823*np.sin(q0)*np.sin(q4)*np.sin(q1 + q2 + q3) + 0.39225*np.sin(q0)*np.sin(q1 + q2)  + 0.09465*np.sin(q0)*np.cos(q1 + q2 + q3) , -0.0823*np.sin(q4)*np.cos(q1 + q2 + q3)  + 0.09465*np.sin(q1 + q2 + q3) - 0.425*np.cos(q1) - 0.39225*np.cos(q1 + q2), 1, 0, 0], 
                       [ 0.0823*np.sin(q4)*np.sin(q1 + q2 + q3)*np.cos(q0) + 0.39225*np.sin(q1 + q2)*np.cos(q0)  + 0.09465*np.cos(q0)*np.cos(q1 + q2 + q3), 0.0823*np.sin(q0)*np.sin(q4)*np.sin(q1 + q2 + q3) + 0.39225*np.sin(q0)*np.sin(q1 + q2)  + 0.09465*np.sin(q0)*np.cos(q1 + q2 + q3) , -0.0823*np.sin(q4)*np.cos(q1 + q2 + q3)  + 0.09465*np.sin(q1 + q2 + q3) - 0.39225*np.cos(q1 + q2), 1, 0, 0], 
                       [(0.0823*np.sin(q4)*np.sin(q1 + q2 + q3)  + 0.09465*np.cos(q1 + q2 + q3))*np.cos(q0), (0.0823*np.sin(q4)*np.sin(q1 + q2 + q3) + 0.09465*np.cos(q1 + q2 + q3))*np.sin(q0), -0.0823*np.sin(q4)*np.cos(q1 + q2 + q3) + 0.09465*np.sin(q1 + q2 + q3), 1, 0, 0], 
                       [-0.0823*(1.0*np.sin(q0) )*np.sin(q4) - 0.0823*np.cos(q0)*np.cos(q4)*np.cos(q1 + q2 + q3), 0.0823*( 1.0*np.cos(q0))*np.sin(q4) - 0.0823*np.sin(q0)*np.cos(q4)*np.cos(q1 + q2 + q3),  - 0.0823*np.sin(q1 + q2 + q3)*np.cos(q4), 0, 0, 1], 
                       [0, 0, 0, 1, 0, 0]])
    elif str(rob).lower()=='ur10':
        J = np.matrix([[0.0922*np.sin(q0)*np.sin(q4)*np.cos(q1 + q2 + q3)  - 0.1157*np.sin(q0)*np.sin(q1 + q2 + q3) + 0.612*np.sin(q0)*np.cos(q1) + 0.5723*np.sin(q0)*np.cos(q1 + q2)  + 0.0922*np.cos(q0)*np.cos(q4) + 0.163941*np.cos(q0),  0.0922*np.sin(q0)*np.cos(q4) + 0.163941*np.sin(q0) - 0.0922*np.sin(q4)*np.cos(q0)*np.cos(q1 + q2 + q3)  + 0.1157*np.sin(q1 + q2 + q3)*np.cos(q0) - 0.612*np.cos(q0)*np.cos(q1) - 0.5723*np.cos(q0)*np.cos(q1 + q2), 0, 0, 0, 1], 
                       [ 0.612*np.sin(q1)*np.cos(q0) + 0.0922*np.sin(q4)*np.sin(q1 + q2 + q3)*np.cos(q0) + 0.5723*np.sin(q1 + q2)*np.cos(q0)  + 0.1157*np.cos(q0)*np.cos(q1 + q2 + q3), 0.612*np.sin(q0)*np.sin(q1) + 0.0922*np.sin(q0)*np.sin(q4)*np.sin(q1 + q2 + q3) + 0.5723*np.sin(q0)*np.sin(q1 + q2)  + 0.1157*np.sin(q0)*np.cos(q1 + q2 + q3) , -0.0922*np.sin(q4)*np.cos(q1 + q2 + q3)  + 0.1157*np.sin(q1 + q2 + q3) - 0.612*np.cos(q1) - 0.5723*np.cos(q1 + q2), 1, 0, 0], 
                       [ 0.0922*np.sin(q4)*np.sin(q1 + q2 + q3)*np.cos(q0) + 0.5723*np.sin(q1 + q2)*np.cos(q0) + 0.1157*np.cos(q0)*np.cos(q1 + q2 + q3), 0.0922*np.sin(q0)*np.sin(q4)*np.sin(q1 + q2 + q3) + 0.5723*np.sin(q0)*np.sin(q1 + q2)  + 0.1157*np.sin(q0)*np.cos(q1 + q2 + q3) , -0.0922*np.sin(q4)*np.cos(q1 + q2 + q3)  + 0.1157*np.sin(q1 + q2 + q3) - 0.5723*np.cos(q1 + q2), 1, 0, 0], 
                       [(0.0922*np.sin(q4)*np.sin(q1 + q2 + q3)  + 0.1157*np.cos(q1 + q2 + q3))*np.cos(q0), (0.0922*np.sin(q4)*np.sin(q1 + q2 + q3)  + 0.1157*np.cos(q1 + q2 + q3))*np.sin(q0), -0.0922*np.sin(q4)*np.cos(q1 + q2 + q3)  + 0.1157*np.sin(q1 + q2 + q3), 1, 0, 0], 
                       [-0.0922*(1.0*np.sin(q0) )*np.sin(q4) - 0.0922*np.cos(q0)*np.cos(q4)*np.cos(q1 + q2 + q3), 0.0922*( 1.0*np.cos(q0))*np.sin(q4) - 0.0922*np.sin(q0)*np.cos(q4)*np.cos(q1 + q2 + q3),  - 0.0922*np.sin(q1 + q2 + q3)*np.cos(q4), 0, 0, 1], 
                       [0, 0, 0, 1, 0, 0]])
    return J

                    
def RotatMatr2AxisAng(Matrix):
    '''
    Convert the rotation matrix to axis angle
    '''
    R = Matrix
    theta = np.arccos(0.5*(R[0,0]+R[1,1]+R[2,2]-1))  
    e1 = (R[2,1]-R[1,2])/(2*np.sin(theta))
    e2 = (R[0,2]-R[2,0])/(2*np.sin(theta))
    e3 = (R[1,0]-R[0,1])/(2*np.sin(theta))    
    axis_ang = np.array([theta*e1,theta*e2,theta*e3])    
    return axis_ang
    

def AxisAng2RotaMatri(angle_vec):
    '''
    Convert an Axis angle to rotation matrix
    AxisAng2Matrix(angle_vec)
    angle_vec need to be a 3D Axis angle  
    '''
    theta = math.sqrt(angle_vec[0]**2+angle_vec[1]**2+angle_vec[2]**2)
    if theta==0.:
        return np.identity(3, dtype=float)
    
    cs = np.cos(theta)
    si = np.sin(theta)
    e1 = angle_vec[0]/theta
    e2 = angle_vec[1]/theta
    e3 = angle_vec[2]/theta
            
    R=np.zeros((3,3))
    R[0,0] = (1-cs)*e1**2+cs
    R[0,1] = (1-cs)*e1*e2-e3*si
    R[0,2] = (1-cs)*e1*e3+e2*si
    R[1,0] = (1-cs)*e1*e2+e3*si
    R[1,1] = (1-cs)*e2**2+cs
    R[1,2] = (1-cs)*e2*e3-e1*si
    R[2,0] = (1-cs)*e1*e3-e2*si
    R[2,1] = (1-cs)*e2*e3+e1*si
    R[2,2] = (1-cs)*e3**2+cs  
    return R


def Rotat2TransMarix(Rota_Matrix,pose):
    '''
    convert the rotation matrix and pose to transformation matrix
    '''
    tran_mat = np.zeros((4,4))
    tran_mat[:3,:3] = Rota_Matrix[:3,:3]
    tran_mat[3,3] = 1
    tran_mat[:3,3] = pose[:3]
    return tran_mat
        
def Pose2Tran_Mat(pose):
    '''
    Convert an pose to a transformation matrix
    AxisAng2Matrix(pose)
    '''  
    rot_mat = AxisAng2RotaMatri(pose[-3:])
    tran_mat = Rotat2TransMarix(rot_mat,pose)  
    return tran_mat

def Tran_Mat2Pose(Tran_Mat):
    '''
    Convert a transformation matrix to pose 
    '''
    pose = np.zeros([6])
    pose[:3] = Tran_Mat[:3,3]
    Rota_mat = np.zeros([3,3])
    Rota_mat[:3,:3] = Tran_Mat[:3,:3]
    angle_vec = RotatMatr2AxisAng(Rota_mat)
    pose[-3:] = angle_vec
    return pose

def cmpleate_rotation_matrix(start_vector):  
    ''' This function make rotation matrix where the first column is the 
        input vector.
    '''
    
    a = np.matrix(start_vector)
    a = a/ np.linalg.norm(a)   
    U, s, Vh = linalg.svd(a, full_matrices=True)

    Vh[0] = a # to ensure that the virst vector is not -a
    Vh[2] = np.cross(a,Vh[1]) 
    Vh = np.transpose(Vh)
    return np.array(Vh)
   
def Inverse_kin(target_pos,init_joint_pos=[0,0,0,0,0,0],tcpOffset=[0,0,0, 0,0,0]):
    '''
    Find the inverse kinematics
    target_pos = the target pos vector
    init_joint_pos (optional) = the initial joint vector
    '''
    # Define a robot from URDF file
    my_chain = ik.chain.Chain.from_urdf_file('URDF/UR5.URDF')        
    #Convert pos to transfer matrix
    #Mar = Pose2Tran_Mat(target_pos)

    T_target = Pose2Tran_Mat(pose=target_pos)
    T_tcp = Pose2Tran_Mat(pose=tcpOffset)
    Mar = np.matmul(T_target, np.linalg.inv(T_tcp))

    #Inverse kinematics       
    if len(init_joint_pos)<6 :            
        return None
    else:
        joint_init = init_joint_pos.copy()       
    # add a [0] at the base frame
    joint_initadd = np.zeros([7])
    joint_initadd[1:] = joint_init[:]        
    ikin = my_chain.inverse_kinematics(target=Mar,initial_position=joint_initadd.copy())
    
    print('***************************************************************************')
    print('Pose: [{: 06.3f}, {: 06.3f}, {: 06.3f},   {: 06.3f}, {: 06.3f}, {: 06.3f}]'.format(*target_pos))
    print('---------------------------------------------------------------------------')
    print(ikin)
    print('---------------------------------------------------------------------------')
    print('Joints Returned: [{: 06.3f}, {: 06.3f}, {: 06.3f},   {: 06.3f}, {: 06.3f}, {: 06.3f}]'.format(*ikin[1:]))
    print('***************************************************************************')

    
    return ikin[1:]
    
def Forward_kin(joint):
    '''
    Find the forward kinematics 
    '''
    # Define a robot from URDF file
    my_chain = ik.chain.Chain.from_urdf_file('URDF/UR5.URDF')
    # add a [0] in joint anlges, due to the defination of URDF
    joint_new = np.zeros([7])
    joint_new[1:] = joint[:]
    fk = my_chain.forward_kinematics(joint_new)
    #convert transfer matrix to pos
    axis_ang = Tran_Mat2Pose(fk)  
    return axis_ang    


def Vektor_from_Base_to_TCP(vectorBase,axisangle):
    """
    Function to calculate the rotation of a vector with an axis angle. This function can be used to transform a vector from the base coordinate system to the tcp coordinate system. 
    (To go from tcp to Base use -1*le as input)
    
    Parameters:
    vectorBase (np.array, 3*1): the vector to be rotated
    axisangle (np.array, 3*1): the axis angle between the base and the tcp
    """
    # calculating the rotation angle by finding the length of the axis angle
    angle=axisangle[0]*axisangle[0]+axisangle[1]*axisangle[1]+axisangle[2]*axisangle[2]
    angle=np.sqrt(angle)
    # Defining the unit rotation angle
    eRot=axisangle/angle
    # calculation of the tcp vector using Rodrigues' rotation formula
    vectorBase = np.array(vectorBase)
    vectorTCP= np.cos(angle)*vectorBase+np.sin(angle)*np.cross(eRot,vectorBase)+(1-np.cos(angle))*np.dot(eRot,vectorBase)*eRot
    return vectorTCP
    
    