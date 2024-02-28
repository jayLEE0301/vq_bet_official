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

from .. import URBasic

class RobotModel(object):
    '''
    Data class holding all data and states
         
    Input parameters:

    '''


    def __init__(self):
        '''
        Constructor see class description for more info.
        '''
        logger = URBasic.dataLogging.DataLogging()        
        name = logger.AddEventLogging(__name__)        
        self.__logger = logger.__dict__[name]
        self.__logger.info('Init done')
        
        #Universal Robot Model content
        self.password = None
        self.ipAddress = None
        
        self.dataDir = {'timestamp':None,
                         'target_q':None,
                         'target_q':None,
                         'target_qd':None,
                         'target_qdd':None,
                         'target_current':None,
                         'target_moment':None,
                         'actual_q':None,
                         'actual_qd':None,
                         'actual_current':None,
                         'joint_control_output':None,
                         'actual_TCP_pose':None,
                         'actual_TCP_speed':None,
                         'actual_TCP_force':None,
                         'target_TCP_pose':None,
                         'target_TCP_speed':None,
                         'actual_digital_input_bits':None,
                         'joint_temperatures':None,
                         'actual_execution_time':None,
                         'robot_mode':None,
                         'joint_mode':None,
                         'safety_mode':None,
                         'actual_tool_accelerometer':None,
                         'speed_scaling':None,
                         'target_speed_fraction':None,
                         'actual_momentum':None,
                         'actual_main_voltage':None,
                         'actual_robot_voltage':None,
                         'actual_robot_current':None,
                         'actual_joint_voltage':None,
                         'actual_digital_output_bits':None,
                         'runtime_state':None,
                         'robot_status_bits':None,
                         'safety_status_bits':None,
                         'analog_io_types':None,
                         'standard_analog_input0':None,
                         'standard_analog_input1':None,
                         'standard_analog_output0':None,
                         'standard_analog_output1':None,
                         'io_current':None,
                         'euromap67_input_bits':None,
                         'euromap67_output_bits':None,
                         'euromap67_24V_voltage':None,
                         'euromap67_24V_current':None,
                         'tool_mode':None,
                         'tool_analog_input_types':None,
                         'tool_analog_input0':None,
                         'tool_analog_input1':None,
                         'tool_output_voltage':None,
                         'tool_output_current':None,
                         'tcp_force_scalar':None,
                         'output_bit_registers0_to_31':None,
                         'output_bit_registers32_to_63':None,
                         'output_int_register_0':None,
                         'output_int_register_1':None,
                         'output_int_register_2':None,
                         'output_int_register_3':None,
                         'output_int_register_4':None,
                         'output_int_register_5':None,
                         'output_int_register_6':None,
                         'output_int_register_7':None,
                         'output_int_register_8':None,
                         'output_int_register_9':None,
                         'output_int_register_10':None,
                         'output_int_register_11':None,
                         'output_int_register_12':None,
                         'output_int_register_13':None,
                         'output_int_register_14':None,
                         'output_int_register_15':None,
                         'output_int_register_16':None,
                         'output_int_register_17':None,
                         'output_int_register_18':None,
                         'output_int_register_19':None,
                         'output_int_register_20':None,
                         'output_int_register_21':None,
                         'output_int_register_22':None,
                         'output_int_register_23':None,
                         'output_double_register_0':None,
                         'output_double_register_1':None,
                         'output_double_register_2':None,
                         'output_double_register_3':None,
                         'output_double_register_4':None,
                         'output_double_register_5':None,
                         'output_double_register_6':None,
                         'output_double_register_7':None,
                         'output_double_register_8':None,
                         'output_double_register_9':None,
                         'output_double_register_10':None,
                         'output_double_register_11':None,
                         'output_double_register_12':None,
                         'output_double_register_13':None,
                         'output_double_register_14':None,
                         'output_double_register_15':None,
                         'output_double_register_16':None,
                         'output_double_register_17':None,
                         'output_double_register_18':None,
                         'output_double_register_19':None,
                         'output_double_register_20':None,
                         'output_double_register_21':None,
                         'output_double_register_22':None,
                         'output_double_register_23':None,
                         'urPlus_force_torque_sensor':None,
                         'urPlus_totalMovedVerticalDistance':None
                         }
                            
        
        self.rtcConnectionState = None
        self.rtcProgramRunning = False
        self.rtcProgramExecutionError = False
        self.stopRunningFlag = False
        self.forceRemoteActiveFlag = False 

        # UR plus content
        self.hasForceTorqueSensor = False
        self.forceTourqe = None
        
    def RobotTimestamp(self):return self.dataDir['timestamp']
    def LastUpdateTimestamp(self):raise NotImplementedError('Function Not yet implemented')
    def RTDEConnectionState(self):raise NotImplementedError('Function Not yet implemented')
    def RuntimeState(self): return self.rtcProgramRunning
    def StopRunningFlag(self): return self.stopRunningFlag
    def DigitalInputbits(self,n):
        if n>=0 & n<8:
            n = pow(2,n)
            return n&self.dataDir['actual_digital_input_bits']==n
        else:
            return None
        
    def ConfigurableInputBits(self,n):
        if n>=8 & n<16:
            n = pow(2,n+8)
            return n&self.dataDir['actual_digital_input_bits']==n
        else:
            return None
    
    def DigitalOutputBits(self,n):
        if n>=0 & n<8:
            n = pow(2,n)
            return n&self.dataDir['actual_digital_output_bits']==n
        else:
            return None
    
    def ConfigurableOutputBits(self,n):
        if n>=8 & n<16:
            n = pow(2,n+8)
            return n&self.dataDir['actual_digital_output_bits']==n
        else:
            return None
    
    def RTDEProtocolVersion(self):raise NotImplementedError('Function Not yet implemented')
    def ActualTCPPose(self):return self.dataDir['actual_TCP_pose']
    def RobotModee(self):raise NotImplementedError('Function Not yet implemented')
    def SafetyMode(self):raise NotImplementedError('Function Not yet implemented')
    def TargetQ(self):raise NotImplementedError('Function Not yet implemented')
    def TargetQD(self):raise NotImplementedError('Function Not yet implemented')
    def TargetQDD(self):raise NotImplementedError('Function Not yet implemented')
    def TargetCurrent(self):raise NotImplementedError('Function Not yet implemented')
    def TargetMoment(self):raise NotImplementedError('Function Not yet implemented')     
    def ActualQ(self):return self.dataDir['actual_q']
    # def ActualQD(self):raise NotImplementedError('Function Not yet implemented')
    def ActualQD(self):return self.dataDir['actual_qd']
    def ActualCurrent(self):raise NotImplementedError('Function Not yet implemented')
    def JointControlOutput(self):raise NotImplementedError('Function Not yet implemented')
    def ActualTCPSpeed(self):raise NotImplementedError('Function Not yet implemented')
    def ActualTCPForce(self):raise NotImplementedError('Function Not yet implemented')
    def TargetTCPPose(self):raise NotImplementedError('Function Not yet implemented')
    def TargetTCPSpeed(self):raise NotImplementedError('Function Not yet implemented')
    def JointTemperatures(self):raise NotImplementedError('Function Not yet implemented')
    def ActualExecutionTime(self):raise NotImplementedError('Function Not yet implemented')
    def JointMode(self):raise NotImplementedError('Function Not yet implemented')
    def ActualToolAccelerometer(self):raise NotImplementedError('Function Not yet implemented')
    def SpeedScaling(self):raise NotImplementedError('Function Not yet implemented')
    def TargetSpeedFraction(self):raise NotImplementedError('Function Not yet implemented')
    def ActualMomentum(self):raise NotImplementedError('Function Not yet implemented')
    def ActualMainVoltage(self):raise NotImplementedError('Function Not yet implemented')
    def ActualRobotVoltage(self):raise NotImplementedError('Function Not yet implemented')
    def ActualRobotCurrent(self):raise NotImplementedError('Function Not yet implemented')
    def ActualJointVoltage(self):raise NotImplementedError('Function Not yet implemented')
    def RunTimeState(self):raise NotImplementedError('Function Not yet implemented')
    def IoCurrent(self):raise NotImplementedError('Function Not yet implemented')
    def ToolAnalogInput0(self):raise NotImplementedError('Function Not yet implemented')
    def ToolAnalogInput1(self):raise NotImplementedError('Function Not yet implemented')
    def ToolOutputCurrent(self):raise NotImplementedError('Function Not yet implemented')
    def ToolOutputVoltage(self):raise NotImplementedError('Function Not yet implemented')
    def StandardAnalogInput(self,n):
        if n == 0:
            return self.dataDir['standard_analog_input0']
        elif n == 1:
            return self.dataDir['standard_analog_input1']
        else:
            raise KeyError('Index out of range')

    def StandardAnalogOutput(self):raise NotImplementedError('Function Not yet implemented')

    def RobotStatus(self):
        '''
        SafetyStatusBit class defined in the bottom of this file
        '''
        result = RobotStatusBit()
        result.PowerOn            =  1&self.dataDir['robot_status_bits']==1
        result.ProgramRunning     =  2&self.dataDir['robot_status_bits']==2
        result.TeachButtonPressed =  4&self.dataDir['robot_status_bits']==4
        result.PowerButtonPressed =  8&self.dataDir['robot_status_bits']==8
        return result
    
    def SafetyStatus(self):
        '''
        SafetyStatusBit class defined in the bottom of this file
        '''
        result = SafetyStatusBit()
        result.NormalMode             =     1&self.dataDir['safety_status_bits']==1
        result.ReducedMode            =     2&self.dataDir['safety_status_bits']==2
        result.ProtectiveStopped      =     4&self.dataDir['safety_status_bits']==4
        result.RecoveryMode           =     8&self.dataDir['safety_status_bits']==8
        result.SafeguardStopped       =    16&self.dataDir['safety_status_bits']==16
        result.SystemEmergencyStopped =    32&self.dataDir['safety_status_bits']==32
        result.RobotEmergencyStopped  =    64&self.dataDir['safety_status_bits']==64
        result.EmergencyStopped       =   128&self.dataDir['safety_status_bits']==128
        result.Violation              =   256&self.dataDir['safety_status_bits']==256
        result.Fault                  =   512&self.dataDir['safety_status_bits']==512
        result.StoppedDueToSafety     =  1024&self.dataDir['safety_status_bits']==1024
        return result
    
    def TcpForceScalar(self):raise NotImplementedError('Function Not yet implemented')
    
    def OutputBitRegister(self):
        result = [None]*64
        for ii in range(64):
            if ii<32 and self.dataDir['output_bit_registers0_to_31'] is not None:
                result[ii] = 2**(ii)&self.dataDir['output_bit_registers0_to_31']==2**(ii)
            elif ii>31 and self.dataDir['output_bit_registers32_to_63'] is not None:
                result[ii] = 2**(ii-32)&self.dataDir['output_bit_registers32_to_63']==2**(ii-32)
        return result
    
    def OutputDoubleRegister(self):raise NotImplementedError('Function Not yet implemented')
    def UrControlVersion(self):raise NotImplementedError('Function Not yet implemented')
    def ClearToSend(self):raise NotImplementedError('Function Not yet implemented')


class RobotStatusBit(object):
    PowerOn = None
    ProgramRunning = None
    TeachButtonPressed = None
    PowerButtonPressed = None

class SafetyStatusBit(object):
    NormalMode = None
    ReducedMode = None
    ProtectiveStopped = None
    RecoveryMode = None
    SafeguardStopped = None
    SystemEmergencyStopped = None
    RobotEmergencyStopped = None
    EmergencyStopped = None
    Violation = None
    Fault = None
    StoppedDueToSafety = None
    