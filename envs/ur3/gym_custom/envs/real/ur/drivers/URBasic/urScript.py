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
import ctypes
__author__ = "Martin Huus Bjerge"
__copyright__ = "Copyright 2017, Rope Robotics ApS, Denmark"
__license__ = "MIT License"

from .. import URBasic
import numpy as np
import time

class UrScript(object):
    '''
    Interface to remote access UR script commands.
    For more details see the script manual at this site:
    http://www.universal-robots.com/download/
    
    Beside the implementation of the script interface, this class also inherits from the 
    Real Time Client and RTDE interface and thereby also open a connection to these data interfaces.
    The Real Time Client in this version is only used to send program and script commands 
    to the robot, not to read data from the robot, all data reading is done via the RTDE interface.
    
    The constructor takes a UR robot hostname as input, and a RTDE configuration file, and optional a logger object.

    Input parameters:
    host (string):  hostname or IP of UR Robot (RT CLient server)
    rtde_conf_filename (string):  Path to xml file describing what channels to activate
    logger (URBasis_DataLogging obj): A instance if a logger object if common logging is needed.

    
    Example:
    rob = URBasic.urScript.UrScript('192.168.56.101', rtde_conf_filename='rtde_configuration.xml')
    self.close_rtc()
    '''


    def __init__(self, host, robotModel, hasForceTorque=False):
        '''
        Constructor see class description for more info.
        '''
        logger = URBasic.dataLogging.DataLogging()        
        name = logger.AddEventLogging(__name__)        
        self.__logger = logger.__dict__[name]
        self.robotConnector = URBasic.robotConnector.RobotConnector(robotModel, host, hasForceTorque)
        #time.sleep(200)
        while(self.robotConnector.RobotModel.ActualTCPPose() is None):      ## check paa om vi er startet
            print("waiting for everything to be ready")
            time.sleep(1)
        self.__logger.info('Init done')
#############   Module motion   ###############

    def waitRobotIdleOrStopFlag(self):
    
        while(self.robotConnector.RobotModel.RuntimeState() and not self.robotConnector.RobotModel.StopRunningFlag()):
            time.sleep(0.002)

        if self.robotConnector.RobotModel.rtcProgramExecutionError:
            raise RuntimeError('Robot program execution error!!!')
        
    def movej(self, q=None, a=1.4, v =1.05, t =0, r =0, wait=True, pose=None):
        '''
        Move to position (linear in joint-space) When using this command, the
        robot must be at standstill or come from a movej og movel with a
        blend. The speed and acceleration parameters controls the trapezoid
        speed profile of the move. The $t$ parameters can be used in stead to
        set the time for this move. Time setting has priority over speed and
        acceleration settings. The blend radius can be set with the $r$
        parameters, to avoid the robot stopping at the point. However, if he
        blend region of this mover overlaps with previous or following regions,
        this move will be skipped, and an 'Overlapping Blends' warning
        message will be generated.
        Parameters:
        q:    joint positions (Can also be a pose)
        a:    joint acceleration of leading axis [rad/s^2]
        v:    joint speed of leading axis [rad/s]
        t:    time [S]
        r:    blend radius [m]
        wait: function return when movement is finished
        pose: target pose
        '''
        prg =  '''def move_j():
{movestr}
end
'''
        movestr = self._move(movetype='j', pose=pose, a=a, v=v, t=t, r=r, wait=wait, q=q)
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.SendProgram(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        
    def movel(self, pose=None, a=1.2, v =0.25, t =0, r =0, wait=True, q=None):
        '''
        Move to position (linear in tool-space)
        See movej.
        Parameters:
        pose: target pose (Can also be a joint position)
        a:    tool acceleration [m/s^2]
        v:    tool speed [m/s]
        t:    time [S]
        r:    blend radius [m]
        wait: function return when movement is finished
        q:    joint position
        '''

        prg =  '''def move_l():
{movestr}
end
'''
        movestr = self._move(movetype='l', pose=pose, a=a, v=v, t=t, r=r, wait=wait, q=q)
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.SendProgram(programString)
        #time.sleep(0.5)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        
        

    def movep(self, pose=None, a=1.2, v =0.25, r =0, wait=True, q=None):
        '''
        Move Process
        
        Blend circular (in tool-space) and move linear (in tool-space) to
        position. Accelerates to and moves with constant tool speed v.
        Parameters:
        pose: list of target pose (pose can also be specified as joint
              positions, then forward kinematics is used to calculate the corresponding pose)
        a:    tool acceleration [m/s^2]
        v:    tool speed [m/s]
        r:    blend radius [m]
        wait: function return when movement is finished
        q:    list of target joint positions  
        '''

        prg =  '''def move_p():
{movestr}
end
'''
        movestr = self._move(movetype='p', pose=pose, a=a, v=v, t=0, r=r, wait=wait, q=q)
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.SendProgram(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        
        
    def movec(self, pose_via=None, pose_to=None, a=1.2, v =0.25, r =0, wait=True, q_via=None, q_to=None):
        '''
        Move Circular: Move to position (circular in tool-space)

        TCP moves on the circular arc segment from current pose, through pose via to pose to. 
        Accelerates to and moves with constant tool speed v.

        Parameters:
        pose_via: path point (note: only position is used). (pose via can also be specified as joint positions,
                  then forward kinematics is used to calculate the corresponding pose)
        pose_to:  target pose (pose to can also be specified as joint positions, then forward kinematics 
                  is used to calculate the corresponding pose)
        a:        tool acceleration [m/s^2]
        v:        tool speed [m/s]
        r:        blend radius (of target pose) [m]
        wait:     function return when movement is finished
        q_via:    list of via joint positions
        q_to:     list of target joint positions
        '''

        prg =  '''def move_p():
{movestr}
end
'''
        movestr = self._move(movetype='p', pose=pose_to, a=a, v=v, t=0, r=r, wait=wait, q=q_to,pose_via=pose_via, q_via=q_via)
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.SendProgram(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        
        
 
    def _move(self, movetype, pose=None, a=1.2, v=0.25, t=0, r=0, wait=True, q=None, pose_via=None, q_via=None):
        '''
        General move Process
        
        Blend circular (in tool-space) and move linear (in tool-space) to
        position. Accelerates to and moves with constant tool speed v.
        Parameters:
        movetype: j, l, p, c
        pose: list of target pose (pose can also be specified as joint
              positions, then forward kinematics is used to calculate the corresponding pose)
        a:    tool acceleration [m/s^2]
        v:    tool speed [m/s]
        r:    blend radius [m]
        wait: function return when movement is finished
        q:    list of target joint positions  
        '''

        prefix="p"
        t_val=''
        pose_via_val=''
        if pose is None:
            prefix=""
            pose=q
        pose = np.array(pose)
        if movetype == 'j' or movetype == 'l':
            tval='t={t},'.format(**locals())
        
        if movetype =='c':
            if pose_via is None:
                prefix_via=""
                pose_via=q_via
            else:
                prefix_via="p"
            
            pose_via = np.array(pose_via)
            
            #Check if pose and pose_via have same shape 
            if (pose.shape != pose_via.shape):
                return False
        
        movestr = ''
        if np.size(pose.shape)==2:
            for idx in range(np.size(pose, 0)):
                posex = np.round(pose[idx], 4)
                posex = posex.tolist()
                if movetype =='c':
                    pose_via_x = np.round(pose_via[idx], 4)
                    pose_via_x = pose_via_x.tolist()
                    pose_via_val='{prefix_via}{pose_via_x},'
                    
                if (np.size(pose, 0)-1)==idx:
                    r=0
                movestr +=  '    move{movetype}({pose_via_val} {prefix}{posex}, a={a}, v={v}, {t_val} r={r})\n'.format(**locals())
                
            movestr +=  '    stopl({a})\n'.format(**locals())
        else:
            posex = np.round(pose, 4)
            posex = posex.tolist()
            if movetype =='c':
                pose_via_x = np.round(pose_via, 4)
                pose_via_x = pose_via_x.tolist()
                pose_via_val='{prefix_via}{pose_via_x},'
            movestr +=  '    move{movetype}({pose_via_val} {prefix}{posex}, a={a}, v={v}, {t_val} r={r})\n'.format(**locals())
            

        
        return movestr
 
    def force_mode(self, task_frame=[0.,0.,0., 0.,0.,0.], selection_vector=[0,0,1,0,0,0], wrench=[0.,0.,0., 0.,0.,0.], f_type=2, limits=[2, 2, 1.5, 1, 1, 1], wait=False, timeout=60):
        '''
        Set robot to be controlled in force mode
        
        Parameters:
        task frame: A pose vector that defines the force frame relative to the base frame.
        
        selection vector: A 6d vector that may only contain 0 or 1. 1 means that the robot will be
                          compliant in the corresponding axis of the task frame, 0 means the robot is
                          not compliant along/about that axis.

        wrench: The forces/torques the robot is to apply to its environment. These values
                have different meanings whether they correspond to a compliant axis or not.
                Compliant axis: The robot will adjust its position along/about the axis in order
                to achieve the specified force/torque. Non-compliant axis: The robot follows
                the trajectory of the program but will account for an external force/torque
                of the specified value.

        f_type: An integer specifying how the robot interprets the force frame. 
                1: The force frame is transformed in a way such that its y-axis is aligned with a vector
                   pointing from the robot tcp towards the origin of the force frame. 
                2: The force frame is not transformed. 
                3: The force frame is transformed in a way such that its x-axis is the projection of
                   the robot tcp velocity vector onto the x-y plane of the force frame. 
                All other values of f_type are invalid.

        limits: A 6d vector with float values that are interpreted differently for
                compliant/non-compliant axes: 
                Compliant axes: The limit values for compliant axes are the maximum
                                allowed tcp speed along/about the axis. 
                Non-compliant axes: The limit values for non-compliant axes are the
                                    maximum allowed deviation along/about an axis between the
                                    actual tcp position and the one set by the program.
                                    
        '''
        prg = '''def ur_force_mode():
        while True:
            force_mode(p{task_frame}, {selection_vector}, {wrench}, {f_type}, {limits})
            sync()
        end
end
'''
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.SendProgram(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
 
    def end_force_mode(self, wait=False):
        '''
        Resets the robot mode from force mode to normal operation.
        This is also done when a program stops.
        '''
        prg = 'end_force_mode()\n'        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)                      ##### ToDo - check if send or sendprogram
        if(wait):
            self.waitRobotIdleOrStopFlag()
        time.sleep(0.05)
        
    def servoc(self, pose, a=1.2, v =0.25, r =0, wait=True):
        '''
        Servo Circular
        Servo to position (circular in tool-space). Accelerates to and moves with constant tool speed v.
        
        Parameters:
        pose: target pose
        a:    tool acceleration [m/s^2]
        v:    tool speed [m/s]
        r:    blend radius (of target pose) [m]
        '''
        prg = 'servoc(p{pose}, {a}, {v}, {r})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

    def servoj(self, q, t =0.008, lookahead_time=0.1, gain=100, wait=True):
        '''
        Servo to position (linear in joint-space)
        Servo function used for online control of the robot. The lookahead time
        and the gain can be used to smoothen or sharpen the trajectory.
        Note: A high gain or a short lookahead time may cause instability.
        Prefered use is to call this function with a new setpoint (q) in each time
        step (thus the default t=0.008)
        Parameters:
        q:              joint positions [rad]
        t:              time where the command is controlling
                        the robot. The function is blocking for time t [S]
        lookahead_time: time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time
        gain:           proportional gain for following target position, range [100,2000]
        '''
        prg = 'servoj({q}, 0.5, 0.5, {t}, {lookahead_time}, {gain})\n'
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

    def speedj(self, qd, a, t , wait=True):
        '''
        Joint speed
        Accelerate linearly in joint space and continue with constant joint
        speed. The time t is optional; if provided the function will return after
        time t, regardless of the target speed has been reached. If the time t is
        not provided, the function will return when the target speed is reached.
        Parameters:
        qd: joint speeds [rad/s]
        a:  joint acceleration [rad/s^2] (of leading axis)
        t:  time [s] before the function returns (optional)
        '''
        prg = 'speedj({qd}, {a}, {t})\n'
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

    def stopj(self, a, wait=True):
        '''
        Stop (linear in joint space)
        Decellerate joint speeds to zero
        Parameters
        a: joint acceleration [rad/s^2] (of leading axis)
        '''
        prg = 'stopj({a})\n'
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()    
        
    def speedl(self, xd, a=1.4, t=0, aRot=None, wait=True):
        '''
        Tool speed
        Accelerate linearly in Cartesian space and continue with constant tool
        speed. The time t is optional; if provided the function will return after
        time t, regardless of the target speed has been reached. If the time t is
        not provided, the function will return when the target speed is reached.
        Parameters:
        xd:   tool speed [m/s] (spatial vector)
        a:    tool position acceleration [m/s^2]
        t:    time [s] before function returns (optional)
        aRot: tool acceleration [rad/s^2] (optional), if not defined a, position acceleration, is used
        '''
        if aRot is None:
            aRot=a
        prg = '''def ur_speedl():
    while(True):
        speedl({xd}, {a}, {t}, {aRot})
    end
end
'''
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.SendProgram(programString)
#         prg = 'speedl({xd}, {a}, {t}, {aRot})\n'
#         programString = prg.format(**locals())
#         
#         self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

    def stopl(self, a=0.5, wait=True):
        '''
        Stop (linear in tool space)
        Decellerate tool speed to zero
        Parameters:
        a:    tool accleration [m/s^2]
        '''
        prg = 'stopl({a})\n'
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

    def freedrive_mode(self, wait=False):
        '''
        Set robot in freedrive mode. In this mode the robot can be moved around by hand in the 
        same way as by pressing the "freedrive" button.
        The robot will not be able to follow a trajectory (eg. a movej) in this mode.
        '''
        prg = '''def ur_freedrive_mode():
    while(True):
        freedrive_mode()
        sleep(600)
    end
end
'''
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.SendProgram(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
            
    def end_freedrive_mode(self, wait=True):
        '''
        Set robot back in normal position control mode after freedrive mode.
        '''
        prg = 'end_freedrive_mode()\n'        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        time.sleep(0.05)
        
    def teach_mode(self, wait=True):
        '''
        Set robot in freedrive mode. In this mode the robot can be moved
        around by hand in the same way as by pressing the "freedrive" button.
        The robot will not be able to follow a trajectory (eg. a movej) in this mode.
        '''
        prg = '''def ur_teach_mode():
    while True:
        teach_mode()
    end
end
'''
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.SendProgram(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

    def end_teach_mode(self, wait=True):
        '''
        Set robot back in normal position control mode after freedrive mode.
        '''
        prg = 'end_teach_mode()\n'        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        time.sleep(0.05)
        
    def conveyor_pulse_decode(self, in_type, A, B, wait=True):
        '''
        Tells the robot controller to treat digital inputs number A and B as pulses 
        for a conveyor encoder. Only digital input 0, 1, 2 or 3 can be used.

        >>> conveyor pulse decode(1,0,1)

        This example shows how to set up quadrature pulse decoding with
        input A = digital in[0] and input B = digital in[1]

        >>> conveyor pulse decode(2,3)
        
        This example shows how to set up rising and falling edge pulse
        decoding with input A = digital in[3]. Note that you do not have to set
        parameter B (as it is not used anyway).
        Parameters:
            in_type: An integer determining how to treat the inputs on A
                  and B
                  0 is no encoder, pulse decoding is disabled.
                  1 is quadrature encoder, input A and B must be
                    square waves with 90 degree offset. Direction of the
                    conveyor can be determined.
                  2 is rising and falling edge on single input (A).
                  3 is rising edge on single input (A).
                  4 is falling edge on single input (A).

            The controller can decode inputs at up to 40kHz
            A: Encoder input A, values of 0-3 are the digital inputs 0-3.
            B: Encoder input B, values of 0-3 are the digital inputs 0-3.
        '''
        
        prg = 'conveyor_pulse_decode({in_type}, {A}, {B})\n'        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        
    def set_conveyor_tick_count(self, tick_count, absolute_encoder_resolution=0, wait=True):
        '''
        Tells the robot controller the tick count of the encoder. This function is
        useful for absolute encoders, use conveyor pulse decode() for setting
        up an incremental encoder. For circular conveyors, the value must be
        between 0 and the number of ticks per revolution.
        Parameters:
        tick_count: Tick count of the conveyor (Integer)
        absolute_encoder_resolution: Resolution of the encoder, needed to
                                     handle wrapping nicely.
                                     (Integer)
                                    0 is a 32 bit signed encoder, range [-2147483648 ;2147483647] (default)
                                    1 is a 8 bit unsigned encoder, range [0 ; 255]
                                    2 is a 16 bit unsigned encoder, range [0 ; 65535]
                                    3 is a 24 bit unsigned encoder, range [0 ; 16777215]
                                    4 is a 32 bit unsigned encoder, range [0 ; 4294967295]
        '''
        prg = 'set_conveyor_tick_count({tick_count}, {absolute_encoder_resolution})\n'
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
                
    def get_conveyor_tick_count(self):
        '''
        Tells the tick count of the encoder, note that the controller interpolates tick counts to get 
        more accurate movements with low resolution encoders

        Return Value:
            The conveyor encoder tick count
        '''
        
        prg = '''def ur_get_conveyor_tick_count():
    write_output_float_register(0, get_conveyor_tick_count())
end
'''
        programString = prg.format(**locals())
    
        self.robotConnector.RealTimeClient.SendProgram(programString)
        self.waitRobotIdleOrStopFlag()
        return self.robotConnector.RobotModel.outputDoubleRegister[0]

    def stop_conveyor_tracking(self, a=15, aRot ='a', wait=True):
        '''
        Stop tracking the conveyor, started by track conveyor linear() or
        track conveyor circular(), and decellerate tool speed to zero.
        Parameters:
        a:    tool accleration [m/s^2] (optional)
        aRot: tool acceleration [rad/s^2] (optional), if not defined a, position acceleration, is used
        '''
        prg = 'stop_conveyor_tracking({a}, {aRot})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        
        
    def track_conveyor_circular(self, center, ticks_per_revolution, rotate_tool, wait=True):
        '''
        Makes robot movement (movej() etc.) track a circular conveyor.
        
        >>> track conveyor circular(p[0.5,0.5,0,0,0,0],500.0, false)
        
        The example code makes the robot track a circular conveyor with
        center in p[0.5,0.5,0,0,0,0] of the robot base coordinate system, where
        500 ticks on the encoder corresponds to one revolution of the circular
        conveyor around the center.
        Parameters:
        center:               Pose vector that determines the center the conveyor in the base
                              coordinate system of the robot.
        ticks_per_revolution: How many tichs the encoder sees when the conveyor moves one revolution.
        rotate tool:          Should the tool rotate with the coneyor or stay in the orientation 
                              specified by the trajectory (movel() etc.).
        '''
        prg = 'track_conveyor_circular({center}, {ticks_per_revolution}, {rotate_tool})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        


    def track_conveyor_linear(self, direction, ticks_per_meter, wait=True):
        '''
        Makes robot movement (movej() etc.) track a linear conveyor.
        
        >>> track conveyor linear(p[1,0,0,0,0,0],1000.0)
        
        The example code makes the robot track a conveyor in the x-axis of
        the robot base coordinate system, where 1000 ticks on the encoder
        corresponds to 1m along the x-axis.
        Parameters:
        direction:       Pose vector that determines the direction of the conveyor in the base
                         coordinate system of the robot
        ticks per meter: How many tichs the encoder sees when the conveyor moves one meter
        '''
        prg = 'track_conveyor_linear({direction}, {ticks_per_meter})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

    def position_deviation_warning(self, enabled, threshold =0.8, wait=True):
        '''
        Write a message to the log when the robot position deviates from the target position.
        Parameters:
        enabled:   enable or disable position deviation log messages (Boolean)
        threshold: (optional) should be a ratio in the range ]0;1], where 0 is no position deviation and 1 is the
                   position deviation that causes a protective stop (Float).
        '''
        prg = 'position_deviation_warning({enabled}, {threshold})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        
    def reset_revolution_counter(self, qNear=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], wait=True):
        '''
        Reset the revolution counter, if no offset is specified. This is applied on
        joints which safety limits are set to "Unlimited" and are only applied
        when new safety settings are applied with limitted joint angles.

        >>> reset revolution counter()

        Parameters:
        qNear: Optional parameter, reset the revolution counter to one close to the given qNear joint vector. 
               If not defined, the joint's actual number of revolutions are used.
        ''' 
        prg = 'reset_revolution_counter(qNear)\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        
    def set_pos(self, q, wait=True):
        '''
        Set joint positions of simulated robot
        Parameters
        q: joint positions
        '''
        prg = 'set_pos({q})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

####################   Module internals    ####################
    
    def force(self, wait=True):
        '''
        Returns the force exerted at the TCP
        
        Return the current externally exerted force at the TCP. The force is the
        norm of Fx, Fy, and Fz calculated using get tcp force().
        Return Value
        The force in Newtons (float)
        '''
        raise NotImplementedError('Function Not yet implemented')

        
    def get_actual_joint_positions(self, wait=True):
        '''
        Returns the actual angular positions of all joints
        
        The angular actual positions are expressed in radians and returned as a
        vector of length 6. Note that the output might differ from the output of
        get target joint positions(), especially durring acceleration and heavy
        loads.
        
        Return Value:
        The current actual joint angular position vector in rad : [Base,
        Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
        '''
        if(wait):
            self.sync()
        return self.robotConnector.RobotModel.ActualQ() 
        c_pose = self.robotConnector.RobotModel.ActualQ
        
        pose = []
        pose.append(ctypes.c_double(c_pose[0]).value)
        pose.append(ctypes.c_double(c_pose[1]).value)
        pose.append(ctypes.c_double(c_pose[2]).value)
        pose.append(ctypes.c_double(c_pose[3]).value)
        pose.append(ctypes.c_double(c_pose[4]).value)
        pose.append(ctypes.c_double(c_pose[5]).value)
        return pose


        
    def get_actual_joint_speeds(self, wait=True):
        '''
        Returns the actual angular velocities of all joints
        
        The angular actual velocities are expressed in radians pr. second and
        returned as a vector of length 6. Note that the output might differ from
        the output of get target joint speeds(), especially durring acceleration
        and heavy loads.
        
        Return Value
        The current actual joint angular velocity vector in rad/s:
        [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
        '''
        if(wait):
            self.sync()
        return self.robotConnector.RobotModel.ActualQD()

        
    def get_actual_tcp_pose(self, wait=True):
        '''
        Returns the current measured tool pose
        
        Returns the 6d pose representing the tool position and orientation
        specified in the base frame. The calculation of this pose is based on
        the actual robot encoder readings.
        
        Return Value
        The current actual TCP vector : ([X, Y, Z, Rx, Ry, Rz])
        '''
        if(wait):
            self.sync()
        return self.robotConnector.RobotModel.ActualTCPPose()
        c_pose = self.robotConnector.RobotModel.ActualTCPPose
        
        pose = []
        pose.append(ctypes.c_double(c_pose[0]).value)
        pose.append(ctypes.c_double(c_pose[1]).value)
        pose.append(ctypes.c_double(c_pose[2]).value)
        pose.append(ctypes.c_double(c_pose[3]).value)
        pose.append(ctypes.c_double(c_pose[4]).value)
        pose.append(ctypes.c_double(c_pose[5]).value)
        return pose
       
        
    def get_actual_tcp_speed(self,wait=True):
        '''
        Returns the current measured TCP speed
        
        The speed of the TCP retuned in a pose structure. The first three values
        are the cartesian speeds along x,y,z, and the last three define the
        current rotation axis, rx,ry,rz, and the length |rz,ry,rz| defines the angular
        velocity in radians/s.
        Return Value
        The current actual TCP velocity vector; ([X, Y, Z, Rx, Ry, Rz])
        '''
        if(wait):
            self.sync()
        return self.robotConnector.RobotModel.ActualTCPSpeed
        
    def get_actual_tool_flange_pose(self):
        '''
        Returns the current measured tool flange pose
        
        Returns the 6d pose representing the tool flange position and
        orientation specified in the base frame, without the Tool Center Point
        offset. The calculation of this pose is based on the actual robot
        encoder readings.
        
        Return Value:
        The current actual tool flange vector : ([X, Y, Z, Rx, Ry, Rz])
        
        Note: See get actual tcp pose for the actual 6d pose including TCP offset.
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def get_controller_temp(self):
        '''
        Returns the temperature of the control box
        
        The temperature of the robot control box in degrees Celcius.
        
        Return Value:
        A temperature in degrees Celcius (float)
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def get_inverse_kin(self, x, qnear =[-1.6, -1.7, -2.2, -0.8, 1.6, 0.0], maxPositionError =0.0001, maxOrientationError =0.0001):
        '''
        Inverse kinematic transformation (tool space -> joint space). 
        Solution closest to current joint positions is returned, unless qnear defines one.
        
        Parameters:
        x:                   tool pose (spatial vector)
        qnear:               joint positions to select solution. 
                             Optional.
        maxPositionError:    Define the max allowed position error. 
                             Optional.
        maxOrientationError: Define the max allowed orientation error. 
                             Optional.
        
        Return Value:
        joint positions        
        '''
        raise NotImplementedError('Function Not yet implemented')

    def get_joint_temp(self,j):
        '''
        Returns the temperature of joint j
        
        The temperature of the joint house of joint j, counting from zero. j=0 is
        the base joint, and j=5 is the last joint before the tool flange.
        
        Parameters:
        j: The joint number (int)
        
        Return Value:
        A temperature in degrees Celcius (float)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def get_joint_torques(self):
        '''
        Returns the torques of all joints
        
        The torque on the joints, corrected by the torque
        robot itself (gravity, friction, etc.), returned as
        
        Return Value:
        The joint torque vector in ; ([float])
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def get_target_joint_positions(self):
        '''
        Returns the desired angular position of all joints
        
        The angular target positions are expressed in radians and returned as a
        vector of length 6. Note that the output might differ from the output of
        get actual joint positions(), especially durring acceleration and heavy
        loads.
        
        Return Value:
        The current target joint angular position vector in rad: [Base,
        Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def get_target_joint_speeds(self):
        '''
        Returns the desired angular velocities of all joints
        
        The angular target velocities are expressed in radians pr. second and
        returned as a vector of length 6. Note that the output might differ from
        the output of get actual joint speeds(), especially durring acceleration
        and heavy loads.
        
        Return Value:
        The current target joint angular velocity vector in rad/s:
        [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def get_target_tcp_pose(self):
        '''
        Returns the current target tool pose
        
        Returns the 6d pose representing the tool position and orientation
        specified in the base frame. The calculation of this pose is  based on
        the current target joint positions.
        
        Return Value:
        The current target TCP vector; ([X, Y, Z, Rx, Ry, Rz])
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def get_target_tcp_speed(self):
        '''
        Returns the current target TCP speed
        
        The desired speed of the TCP returned in a pose structure. The first
        three values are the cartesian speeds along x,y,z, and the last three
        define the current rotation axis, rx,ry,rz, and the length |rz,ry,rz| defines
        the angular velocity in radians/s.
        
        Return Value:
        The TCP speed; (pose)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def get_tcp_force(self):
        '''
        Returns the wrench (Force/Torque vector) at the TCP
        
        The external wrench is computed based on the error between the joint
        torques required to stay on the trajectory and the expected joint
        torques. The function returns "p[Fx (N), Fy(N), Fz(N), TRx (Nm), TRy (Nm),
        TRz (Nm)]". where Fx, Fy, and Fz are the forces in the axes of the robot
        base coordinate system measured in Newtons, and TRx, TRy, and TRz
        are the torques around these axes measured in Newton times Meters.
        
        Return Value:
        the wrench (pose)
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def get_tool_accelerometer_reading(self):
        '''
        Returns the current reading of the tool accelerometer as a
        three-dimensional vector.
        
        The accelerometer axes are aligned with the tool coordinates, and
        pointing an axis upwards results in a positive reading.
        
        Return Value:
        X, Y, and Z composant of the measured acceleration in
        SI-units (m/s^2).
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def get_tool_current(self):
        '''
        Returns the tool current
        
        The tool current consumption measured in ampere.
        
        Return Value:
        The tool current in ampere.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def is_steady(self):
        '''
        Checks if robot is fully at rest.
        
        True when the robot is fully at rest, and ready to accept higher external
        forces and torques, such as from industrial screwdrivers. It is useful in
        combination with the GUI's wait node, before starting the screwdriver
        or other actuators influencing the position of the robot.
        
        Note: This function will always return false in modes other than the
        standard position mode, e.g. false in force and teach mode.
        
        Return Value:
        True when the robot is fully at rest. Returns False otherwise
        (bool)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def is_within_safety_limits(self, pose):
        '''
        Checks if the given pose is reachable and within the current safety
        limits of the robot.
        
        This check considers joint limits (if the target pose is specified as joint
        positions), safety planes limits, TCP orientation deviation limits and
        range of the robot. If a solution is found when applying the inverse
        kinematics to the given target TCP pose, this pose is considered
        reachable.
        
        Parameters:
        pose: Target pose (which can also be specified as joint positions)
        
        Return Value:
        True if within limits, false otherwise (bool)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def popup(self, s, title='Popup', warning=False, error =False):
        '''
        Display popup on GUI
        
        Display message in popup window on GUI.
        
        Parameters:
        s:       message string
        title:   title string
        warning: warning message?
        error:   error message?
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def powerdown(self):
        '''
        Shutdown the robot, and power off the robot and controller.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_gravity(self, d, wait=True):
        '''
        Set the direction of the acceleration experienced by the robot. When
        the robot mounting is fixed, this corresponds to an accleration of g
        away from the earth's centre.
        
        >>> set gravity([0, 9.82*sin(theta), 9.82*cos(theta)])
        
        will set the acceleration for a robot that is rotated "theta" radians
        around the x-axis of the robot base coordinate system
        
        Parameters:
        d: 3D vector, describing the direction of the gravity, relative to the base of the robot.
        
        Exampel:
        set_gravity([0,0,9.82])  #Robot mounted at flore
        '''
    
        prg = 'set_gravity({d})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()    

    def set_payload(self, m, CoG):
        '''
        Set payload mass and center of gravity
        
        Alternatively one could use set payload mass and set payload cog.
        
        Sets the mass and center of gravity (abbr. CoG) of the payload.
        
        This function must be called, when the payload weight or weight
        distribution changes - i.e when the robot picks up or puts down a
        heavy workpiece.
        
        The CoG argument is optional - if not provided, the Tool Center Point
        (TCP) will be used as the Center of Gravity (CoG). If the CoG argument
        is omitted, later calls to set tcp(pose) will change CoG to the new TCP.
        
        The CoG is specified as a vector, [CoGx, CoGy, CoGz], displacement,
        from the toolmount.
        
        Parameters:
        m:   mass in kilograms
        CoG: Center of Gravity: [CoGx, CoGy, CoGz] in meters.
             Optional.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_payload_cog(self, CoG, wait=True):
        '''
        Set center of gravity
        
        See also set payload.
        
        Sets center of gravity (abbr. CoG) of the payload.
        
        This function must be called, when the weight distribution changes - i.e
        when the robot picks up or puts down a heavy workpiece.
        
        The CoG is specified as a vector, [CoGx, CoGy, CoGz], displacement,
        from the toolmount.
        
        Parameters:
        CoG: Center of Gravity: [CoGx, CoGy, CoGz] in meters.
        '''
        
        prg = 'set_payload_cog({CoG})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

        
                    
    def set_payload_mass(self, m, wait=True):
        '''
        Set payload mass
        
        See also set payload.
        
        Sets the mass of the payload.
        
        This function must be called, when the payload weight changes - i.e
        when the robot picks up or puts down a heavy workpiece.
        
        Parameters:
        m: mass in kilograms
        '''
        prg = 'set_payload_mass({m})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()

    def set_tcp(self, pose, wait=True):
        '''
        Set the Tool Center Point
        
        Sets the transformation from the output flange coordinate system to
        the TCP as a pose.
        
        Parameters:
        pose: A pose describing the transformation.
        '''
        
        if type(pose).__module__ == np.__name__:
            pose = pose.tolist()
        prg = 'set_tcp(p{pose})\n'
        
        programString = prg.format(**locals())
        
        self.robotConnector.RealTimeClient.Send(programString)
        if(wait):
            self.waitRobotIdleOrStopFlag()
        time.sleep(0.05)

    def sleep(self, t):
        '''
        Sleep for an amount of time
        
        Parameters:
        t: time [s]
        '''
        time.sleep(t)
    
    def sync(self):
        '''
        Uses up the remaining "physical" time a thread has in the current
        frame/sample.
        '''
        initialRobotTime = self.robotConnector.RobotModel.RobotTimestamp()
        while(self.robotConnector.RobotModel.RobotTimestamp() == initialRobotTime):
            time.sleep(0.001)

    
    def textmsg(self, s1, s2=''):
        '''
        Send text message to log
        
        Send message with s1 and s2 concatenated to be shown on the GUI
        log-tab
        Parameters
        s1: message string, variables of other types (int, bool poses
        etc.) can also be sent
        s2: message string, variables of other types (int, bool poses
        etc.) can also be sent
        '''
        raise NotImplementedError('Function Not yet implemented')
    
############    Module urmath    #################        
    @staticmethod
    def pose_add(p_1, p_2):
        '''
        Pose addition
        
        Both arguments contain three position parameters (x, y, z) jointly called
        P, and three rotation parameters (R x, R y, R z) jointly called R. This
        function calculates the result x 3 as the addition of the given poses as
        follows:
        p 3.P = p 1.P + p 2.P
        p 3.R = p 1.R * p 2.R
        
        Parameters
        p 1: tool pose 1(pose)
        p 2: tool pose 2 (pose)

        Return Value
        Sum of position parts and product of rotation parts (pose)
        '''
        Trans_1 = URBasic.kinematic.Pose2Tran_Mat(p_1)
        Trans_2 = URBasic.kinematic.Pose2Tran_Mat(p_2)
        Trans_3 = np.matmul(Trans_1, Trans_2)
        p_3 = URBasic.kinematic.Tran_Mat2Pose(Trans_3)
        return p_3
        
        
        
        

############    Module interfaces  #################
        
    def get_configurable_digital_in(self, n):
        '''
        Get configurable digital input signal level
        
        See also get standard digital in and get tool digital in.
        
        Parameters:
        n: The number (id) of the input, integer: [0:7]
        
        Return Value:
        boolean, The signal level. 
        '''
        return self.robotConnector.RobotModel.ConfigurableInputBits(n)
        
    def get_configurable_digital_out(self, n):
        '''
        Get configurable digital output signal level
        
        See also get standard digital out and get tool digital out.
        
        Parameters:
        n: The number (id) of the output, integer: [0:7]
        
        Return Value:
        boolean, The signal level.
        '''
        
        return self.robotConnector.RobotModel.ConfigurableOutputBits(n)
    
    def get_euromap_input(self, port_number):
        '''
        Reads the current value of a specific Euromap67 input signal. See
        http://universal-robots.com/support for signal specifications.
        
        >>> var = get euromap input(3)
        
        Parameters:
        port number: An integer specifying one of the available
        Euromap67 input signals.
        
        Return Value:
        A boolean, either True or False
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def get_euromap_output(self, port_number):
        '''
        Reads the current value of a specific Euromap67 output signal. This
        means the value that is sent from the robot to the injection moulding
        machine. See http://universal-robots.com/support for signal
        specifications.
        
        >>> var = get euromap output(3)
        
        Parameters:
        port number: An integer specifying one of the available
        Euromap67 output signals.
        
        Return Value:
        A boolean, either True or False
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def get_flag(self, n):
        '''
        Flags behave like internal digital outputs. The keep information
        between program runs.
        Parameters
        n: The number (id) of the flag, intereger: [0:32]
        Return Value
        Boolean, The stored bit.
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def get_standard_analog_in(self, n, wait=True):
        '''
        Get standard analog input signal level
        
        See also get tool analog in.
        
        Parameters:
        n: The number (id) of the input, integer: [0:1]
        wait (bool): If True, waits for next data packet before returning. (Default True)
        
        Return Value:
        boolean, The signal level.
        '''
        
        if(wait):
            self.sync()
        return self.robotConnector.RobotModel.StandardAnalogInput(n)
    
    def get_standard_analog_out(self, n, wait=True):
        '''
        Get standard analog output level
        
        Parameters:
        n: The number (id) of the input, integer: [0:1]
        wait (bool): If True, waits for next data packet before returning. (Default True)
        
        Return Value:
        float, The signal level [0;1]
        '''
        if n == 0:
            if(wait):
                self.sync()
            return self.robotConnector.RobotModel.StandardAnalogOutput0
        elif n == 1:
            if(wait):
                self.sync()
                return self.robotConnector.RobotModel.StandardAnalogOutput1
        else:
            raise KeyError('Index out of range')
        
    def get_standard_digital_in(self, n, wait=True):
        '''
        Get standard digital input signal level
        
        See also get configurable digital in and get tool digital in.
        
        Parameters:
        n (int):     The number (id) of the input, integer: [0:7]
        wait (bool): If True, waits for next data packet before returning. (Default True)
        
        Return Value:
        boolean, The signal level.
        '''
        return self.robotConnector.RobotModel.DigitalInputBits(n)
        
    def get_standard_digital_out(self, n):
        '''
        Get standard digital output signal level
        
        See also get configurable digital out and get tool digital out.
        
        Parameters:
        n: The number (id) of the input, integer: [0:7]
        
        Return Value:
        boolean, The signal level.
        '''
        
        return self.robotConnector.RobotModel.DigitalOutputBits(n)
        
    
    def get_tool_analog_in(self, n):
        '''
        Get tool analog input level
        
        See also get standard analog in.
        
        Parameters:
        n: The number (id) of the input, integer: [0:1]
        
        Return Value:
        float, The signal level [0,1]
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def get_tool_digital_in(self, n):
        '''
        Get tool digital input signal level
        
        See also get configurable digital in and
        get standard digital in.
        
        Parameters:
        n: The number (id) of the input, integer: [0:1]
        
        Return Value:
        boolean, The signal level.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def get_tool_digital_out(self, n):
        '''
        Get tool digital output signal level
        
        See also get standard digital out and
        get configurable digital out.
        
        Parameters:
        n: The number (id) of the output, integer: [0:1]
        
        Return Value:
        boolean, The signal level.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def modbus_add_signal(self, IP, slave_number, signal_address, signal_type, signal_name):
        '''
        Adds a new modbus signal for the controller to supervise. Expects no
        response.
        
        >>> modbus add signal("172.140.17.11", 255, 5, 1, "output1")
        Parameters:
        IP:             A string specifying the IP address of the modbus unit 
                        to which the modbus signal is connected.
                        
        slave_number:   An integer normally not used and set to 255, but is a 
                        free choice between 0 and 255.
                        
        signal_address: An integer specifying the address of the either the coil 
                        or the register that this new signal should reflect. 
                        Consult the configuration of the modbus unit for this information.
        
        signal_type:    An integer specifying the type of signal to add. 
                        0 = digital input, 1 = digital output, 
                        2 = register input and 3 = register output.
                        
        signal_name:    A string uniquely identifying the signal. 
                        If a string is supplied which is equal to an already added signal, 
                        the new signal will replace the old one.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def modbus_delete_signal(self, signal_name):
        '''
        Deletes the signal identified by the supplied signal name.
        
        >>> modbus delete signal("output1")
        
        Parameters:
        signal_name: A string equal to the name of the signal that should be deleted.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def modbus_get_signal_status(self, signal_name, is_secondary_program):
        '''
        Reads the current value of a specific signal.
        
        >>> modbus get signal status("output1",False)
        
        Parameters:
        signal name:          A string equal to the name of the signal for which 
                              the value should be gotten.
        
        is_secondary_program: A boolean for interal use only.
                              Must be set to False.
        
        Return Value:
        An integer or a boolean. For digital signals: True or False. For
        register signals: The register value expressed as an unsigned
        integer.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def modbus_send_custom_command(self, IP, slave_number, function_code, data):
        '''
        Sends a command specified by the user to the modbus unit located
        on the specified IP address. Cannot be used to request data, since the
        response will not be received. The user is responsible for supplying data
        which is meaningful to the supplied function code. The builtin function
        takes care of constructing the modbus frame, so the user should not
        be concerned with the length of the command.
        
        >>> modbus send custom command("172.140.17.11",103,6,[17,32,2,88])
        
        The above example sets the watchdog timeout on a Beckhoff BK9050
        to 600 ms. That is done using the modbus function code 6 (preset single
        register) and then supplying the register address in the first two bytes of
        the data array ([17,32] = [0x1120]) and the desired register content in
        the last two bytes ([2,88] = [0x0258] = dec 600).
        
        Parameters:
        IP:            A string specifying the IP address locating the modbus 
                       unit to which the custom command should be send.
        
        slave_number:  An integer specifying the slave number to use for the custom command.
        
        function_code: An integer specifying the function code for the custom command.
        
        data:          An array of integers in which each entry must be a valid byte (0-255) value.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def modbus_set_output_register(self, signal_name, register_value, is_secondary_program):
        '''
        Sets the output register signal identified by the given name to the given
        value.
        
        >>> modbus set output register("output1",300,False)
        
        Parameters:
        signal_name:          A string identifying an output register signal that in advance has been added.
        register_value:       An integer which must be a valid word (0-65535) value.
        is_secondary_program: A boolean for interal use only. Must be set to False.
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def modbus_set_output_signal(self, signal_name, digital_value, is_secondary_program):
        '''
        Sets the output digital signal identified by the given name to the given
        value.
        
        >>> modbus set output signal("output2",True,False)
        
        Parameters:
        signal_name:          A string identifying an output digital signal that in advance has been added.
        digital_value:        A boolean to which value the signal will be set.
        is_secondary_program: A boolean for interal use only. Must be set to False.
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def modbus_set_runstate_dependent_choice(self, signal_name, runstate_choice):
        '''
        Sets whether an output signal must preserve its state from a program,
        or it must be set either high or low when a program is not running.
        
        >>> modbus set runstate dependent choice("output2",1)
        
        Parameters:
        signal_name:     A string identifying an output digital signal that in advance has been added.
        runstate_choice: An integer: 0 = preserve program state, 1 = set low when a program is not running, 2 = set high when a program is not running.
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def modbus_set_signal_update_frequency(self, signal_name, update_frequency):
        '''
        Sets the frequency with which the robot will send requests to the
        Modbus controller to either read or write the signal value.
        
        >>> modbus set signal update frequency("output2",20)
        
        Parameters:
        signal_name: A string identifying an output digital signal that in advance has been added.
        update_frequency: An integer in the range 0-125 specifying the update frequency in Hz.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def read_input_boolean_register(self, address):
        '''
        Reads the boolean from one of the input registers, which can also be
        accessed by a Field bus. Note, uses it's own memory space.
        
        >>> bool val = read input boolean register(3)
        
        Parameters:
        address: Address of the register (0:63)
        
        Return Value:
        The boolean value held by the register (True, False)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def read_input_float_register(self, address):
        '''
        Reads the float from one of the input registers, which can also be
        accessed by a Field bus. Note, uses it's own memory space.
        
        >>> float val = read input float register(3)
        
        Parameters:
        address: Address of the register (0:23)
        
        Return Value:
        The value held by the register (float)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def read_input_integer_register(self, address):
        '''
        Reads the integer from one of the input registers, which can also be
        accessed by a Field bus. Note, uses it's own memory space.
        
        >>> int val = read input integer register(3)
        
        Parameters:
        address: Address of the register (0:23)
        
        Return Value:
        The value held by the register [-2,147,483,648 : 2,147,483,647]
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def read_output_boolean_register(self, address):
        '''
        Reads the boolean from one of the output registers, which can also be
        accessed by a Field bus. Note, uses it's own memory space.
        
        >>> bool val = read output boolean register(3)
        
        Parameters:
        address: Address of the register (0:63)
        
        Return Value:
        The boolean value held by the register (True, False)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def read_output_float_register(self, address):
        '''
        Reads the float from one of the output registers, which can also be
        accessed by a Field bus. Note, uses it's own memory space.
        
        >>> float val = read output float register(3)
        
        Parameters:
        address: Address of the register (0:23)
        
        Return Value:
        The value held by the register (float)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def read_output_integer_register(self, address):
        '''
        Reads the integer from one of the output registers, which can also be
        accessed by a Field bus. Note, uses it's own memory space.
        
        >>> int val = read output integer register(3)
        
        Parameters:
        address: Address of the register (0:23)
        
        Return Value:
        The int value held by the register [-2,147,483,648 :
        2,147,483,647]
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def read_port_bit(self, address):
        '''
        Reads one of the ports, which can also be accessed by Modbus clients
        
        >>> boolval = read port bit(3)
        
        Parameters:
        address: Address of the port (See portmap on Support site,
        page "UsingModbusServer" )
        
        Return Value:
        The value held by the port (True, False)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def read_port_register(self, address):
        '''
        Reads one of the ports, which can also be accessed by Modbus clients
        
        >>> intval = read port register(3)
        
        Parameters:
        address: Address of the port (See portmap on Support site,
        page "UsingModbusServer" )
        
        Return Value:
        The signed integer value held by the port (-32768 : 32767)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def rpc_factory(self, rpcType, url ):
        '''
        Creates a new Remote Procedure Call (RPC) handle. Please read the
        subsection ef{Remote Procedure Call (RPC)} for a more detailed
        description of RPCs.
        
        >>> proxy = rpc factory("xmlrpc", "http://127.0.0.1:8080/RPC2")
        
        Parameters
        rpcType: The type of RPC backed to use. Currently only the "xmlrpc" protocol is available.
        
        url: The URL to the RPC server. Currently two protocols are
        supported: pstream and http. The pstream URL looks
        like "<ip-address>:<port>", for instance
        "127.0.0.1:8080" to make a local connection on port
        8080. A http URL generally looks like
        "http://<ip-address>:<port>/<path>", whereby the
        <path> depends on the setup of the http server. In
        the example given above a connection to a local
        Python webserver on port 8080 is made, which
        expects XMLRPC calls to come in on the path
        "RPC2".
        
        Return Value:
        A RPC handle with a connection to the specified server using
        the designated RPC backend. If the server is not available
        the function and program will fail. Any function that is made
        available on the server can be called using this instance. For
        example "bool isTargetAvailable(int number, ...)" would be
        "proxy.isTargetAvailable(var 1, ...)", whereby any number of
        arguments are supported (denoted by the ...).
        Note: Giving the RPC instance a good name makes programs much
        more readable (i.e. "proxy" is not a very good name).
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def rtde_set_watchdog(self, variable_name, min_frequency, action='pause'):
        '''
        This function will activate a watchdog for a particular input variable to
        the RTDE. When the watchdog did not receive an input update for the
        specified variable in the time period specified by min frequency (Hz),
        the corresponding action will be taken. All watchdogs are removed on
        program stop.
        
        >>> rtde set watchdog("input int register 0", 10, "stop")
        
        Parameters:
        variable name: Input variable name (string), as specified
        by the RTDE interface
        min frequency: The minimum frequency (float) an input
        update is expected to arrive.
        action: Optional: Either "ignore", "pause" or
        "stop" the program on a violation of the
        minimum frequency. The default action is
        "pause".
        
        Return Value:
        None
        Note: Only one watchdog is necessary per RTDE input package to
        guarantee the specified action on missing updates.
        '''
        raise NotImplementedError('Function Not yet implemented')
        
    def set_analog_inputrange(self, port, inputRange):
        '''
        Deprecated: Set range of analog inputs
        
        Port 0 and 1 is in the controller box, 2 and 3 is in the tool connector.
        
        Parameters:
        port: analog input port number, 0,1 = controller, 2,3 = tool
        inputRange: Controller analog input range 0: 0-5V (maps
        automatically onto range 2) and range 2: 0-10V.
        inputRange: Tool analog input range 0: 0-5V (maps
        automatically onto range 1), 1: 0-10V and 2:
        4-20mA.
        Deprecated: The set standard analog input domain and
        set tool analog input domain replace this function. Ports 2-3 should
        be changed to 0-1 for the latter function. This function might be
        removed in the next major release.
        Note: For Controller inputs ranges 1: -5-5V and 3: -10-10V are no longer
        supported and will show an exception in the GUI.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_analog_outputdomain(self, port, domain):
        '''
        Set domain of analog outputs
        
        Parameters:
        port: analog output port number
        domain: analog output domain: 0: 4-20mA, 1: 0-10V
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_configurable_digital_out(self, n, b):
        '''
        Set configurable digital output signal level
        
        See also set standard digital out and set tool digital out.
        
        Parameters:
        n: The number (id) of the output, integer: [0:7]
        b: The signal level. (boolean)
        '''
        #self.robotConnector.RTDE.SetConfigurableDigitalOutput(n, b)
        if b:
            self.robotConnector.RTDE.setData('configurable_digital_output_mask', 2**n)
            self.robotConnector.RTDE.setData('configurable_digital_output', 2**n)
        else:
            self.robotConnector.RTDE.setData('configurable_digital_output_mask', 2**n)
            self.robotConnector.RTDE.setData('configurable_digital_output', 0)
        self.robotConnector.RTDE.sendData()
        self.robotConnector.RTDE.setData('configurable_digital_output_mask', 0)
        self.robotConnector.RTDE.setData('configurable_digital_output', 0)
            
    def set_euromap_output(self, port_number, signal_value):
        '''
        Sets the value of a specific Euromap67 output signal. This means the
        value that is sent from the robot to the injection moulding machine.
        See http://universal-robots.com/support for signal specifications.
        
        >>> set euromap output(3,True)
        
        Parameters:
        port number: An integer specifying one of the available
        Euromap67 output signals.
        signal value: A boolean, either True or False
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_euromap_runstate_dependent_choice(self, port_number, runstate_choice):
        '''
        Sets whether an Euromap67 output signal must preserve its state from a
        program, or it must be set either high or low when a program is not
        running. See http://universal-robots.com/support for signal
        specifications.
        
        >>> set euromap runstate dependent choice(3,0)
        
        Parameters:
        port number: An integer specifying a Euromap67
        output signal.
        runstate choice: An integer: 0 = preserve program state,
        1 = set low when a program is not
        running, 2 = set high when a program is
        not running.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_flag(self, n, b):
        '''
        Flags behave like internal digital outputs. The keep information
        between program runs.
        
        Parameters:
        n: The number (id) of the flag, integer: [0:32]
        b: The stored bit. (boolean)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_runstate_configurable_digital_output_to_value(self, outputId, state):
        '''
        Sets the output signal levels depending on the state of the program
        (running or stopped).
        
        Example: Set configurable digital output 5 to high when program is not
        running.
        
        >>> set runstate configurable digital output to value(5, 2)
        
        Parameters:
        outputId: The output signal number (id), integer: [0:7]
        state: The state of the output, integer: 0 = Preserve
        state, 1 = Low when program is not running, 2 =
        High when program is not running, 3 = High
        when program is running and low when it is
        stopped.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_runstate_standard_analog_output_to_value(self, outputId, state):
        '''
        Sets the output signal levels depending on the state of the program
        (running or stopped).
        
        Example: Set standard analog output 1 to high when program is not
        running.
        
        >>> set runstate standard analog output to value(1, 2)
        
        Parameters:
        outputId: The output signal number (id), integer: [0:1]
        state: The state of the output, integer: 0 = Preserve
        state, 1 = Min when program is not running, 2 =
        Max when program is not running, 3 = Max when
        program is running and Min when it is stopped.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_runstate_standard_digital_output_to_value(self, outputId, state):
        '''
        Sets the output signal levels depending on the state of the program
        (running or stopped).
        
        Example: Set standard digital output 5 to high when program is not
        running.
        
        >>> set runstate standard digital output to value(5, 2)
        Parameters
        outputId: The output signal number (id), integer: [0:7]
        state: The state of the output, integer: 0 = Preserve
        state, 1 = Low when program is not running, 2 =
        High when program is not running, 3 = High
        when program is running and low when it is
        stopped.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_runstate_tool_digital_output_to_value(self, outputId, state):
        '''
        Sets the output signal levels depending on the state of the program
        (running or stopped).
        
        Example: Set tool digital output 1 to high when program is not running.
        
        >>> set runstate tool digital output to value(1, 2)
        
        Parameters:
        outputId: The output signal number (id), integer: [0:1]
        state: The state of the output, integer: 0 = Preserve
        state, 1 = Low when program is not running, 2 =
        High when program is not running, 3 = High
        when program is running and low when it is
        stopped.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_standard_analog_input_domain(self, port, domain):
        '''
        Set domain of standard analog inputs in the controller box
        
        For the tool inputs see set tool analog input domain.
        
        Parameters:
        port: analog input port number: 0 or 1
        domain: analog input domains: 0: 4-20mA, 1: 0-10V
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_standard_analog_out(self, n, f):
        '''
        Set standard analog output level
        Parameters
        n: The number (id) of the input, integer: [0:1]
        f: The relative signal level [0;1] (float)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_standard_digital_out(self, n, b):
        '''
        Set standard digital output signal level
        
        See also set configurable digital out and set tool digital out.
        
        Parameters:
        n: The number (id) of the input, integer: [0:7]
        b: The signal level. (boolean)
        '''
        #self.robotConnector.RTDE.SetStandardDigitalOutput(n, b)
        if b:
            self.robotConnector.RTDE.setData('standard_digital_output_mask', 2**n)
            self.robotConnector.RTDE.setData('standard_digital_output', 2**n)
        else:
            self.robotConnector.RTDE.setData('standard_digital_output_mask', 2**n)
            self.robotConnector.RTDE.setData('standard_digital_output', 0)
        self.robotConnector.RTDE.sendData()
        self.robotConnector.RTDE.setData('standard_digital_output_mask', 0)
        self.robotConnector.RTDE.setData('standard_digital_output', 0)

        
    
    def set_tool_analog_input_domain(self, port, domain):
        '''
        Set domain of analog inputs in the tool
        
        For the controller box inputs see set standard analog input domain.
        
        Parameters:
        port: analog input port number: 0 or 1
        domain: analog input domains: 0: 4-20mA, 1: 0-10V
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_tool_digital_out(self, n, b):
        '''
        Set tool digital output signal level
        
        See also set configurable digital out and
        set standard digital out.
        
        Parameters:
        n: The number (id) of the output, integer: [0:1]
        b: The signal level. (boolean)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def set_tool_voltage(self, voltage):
        '''
        Sets the voltage level for the power supply that delivers power to the
        connector plug in the tool flange of the robot. The votage can be 0, 12
        or 24 volts.
        
        Parameters:
        voltage: The voltage (as an integer) at the tool connector,
        integer: 0, 12 or 24.
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def write_output_boolean_register(self, address, value):
        '''
        Writes the boolean value into one of the output registers, which can
        also be accessed by a Field bus. Note, uses it's own memory space.
        
        >>> write output boolean register(3, True)
        
        Parameters:
        address: Address of the register (0:63)
        value: Value to set in the register (True, False)
        '''
        
    def write_output_float_register(self, address, value):
        '''
        Writes the float value into one of the output registers, which can also
        be accessed by a Field bus. Note, uses it's own memory space.
        
        >>> write output float register(3, 37.68)
        
        Parameters:
        address: Address of the register (0:23)
        value: Value to set in the register (float)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def write_output_integer_register(self, address, value):
        '''
        Writes the integer value into one of the output registers, which can also
        be accessed by a Field bus. Note, uses it's own memory space.
        
        >>> write output integer register(3, 12)
        
        Parameters:
        address: Address of the register (0:23)
        value: Value to set in the register [-2,147,483,648 :
        2,147,483,647]
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def write_port_bit(self, address, value):
        '''
        Writes one of the ports, which can also be accessed by Modbus clients
        
        >>> write port bit(3,True)
        
        Parameters:
        address: Address of the port (See portmap on Support site,
        page "UsingModbusServer" )
        value: Value to be set in the register (True, False)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
    def write_port_register(self, address, value):
        '''
        Writes one of the ports, which can also be accessed by Modbus clients
        
        >>> write port register(3,100)
        
        Parameters:
        address: Address of the port (See portmap on Support site,
        page "UsingModbusServer" )
        value: Value to be set in the port (0 : 65536) or (-32768 :
        32767)
        '''
        raise NotImplementedError('Function Not yet implemented')
    
