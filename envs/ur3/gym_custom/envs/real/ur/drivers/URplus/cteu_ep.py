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

import time
from .. import URBasic
from pymodbus.client.sync import ModbusTcpClient as ModbusClient

class CTEU_EP(object):
    '''
    Controlling the Pneumatics on the robot via a FESTO CTEU-EP that
    is controlling a valveblock - this is via Modbus over TCP
    '''
    def __init__(self, host, robotModel):
        '''
        Constructor - takes ip address to the CTEU-EP box
        '''
        self.__connectionState = URBasic.connectionState.ConnectionState.DISCONNECTED        
        
        logger = URBasic.dataLogging.DataLogging()        
        name = logger.AddEventLogging(__name__)        
        self.__logger = logger.__dict__[name]

        if host is None: #Only for enable code completion
            return

        if(False):
            assert isinstance(robotModel, URBasic.robotModel.RobotModel)  ### This line is to get code completion for RobotModel
        self.__robotModel = robotModel
        
        self.__client = ModbusClient(host=host)
        connected = self.__client.connect()
        if(connected):
            self.__logger.info("Modbus connected to CTEU-EP")
            self.__connectionState = URBasic.connectionState.ConnectionState.CONNECTED        

        else:
            self.__connectionState = URBasic.connectionState.ConnectionState.ERROR
            
        
    def setValve(self, valveNumber, state):
        
        '''
        Set a valve - 0 to 23 to True or False
        '''
        #Valves are 0 to 11 - todo make input validation
        #valveNumber = valveNumber*2
        if not self.__connectionState>URBasic.connectionState.ConnectionState.DISCONNECTED:
            self.__logger.warning('SetValve but not Connected')
            return
             
        if self.__robotModel.StopRunningFlag():
            return
        result = self.__client.write_coil(valveNumber, state)
        if(result == None): #Just one retry
            time.sleep(0.2)
            result = self.__client.write_coil(valveNumber, state)
        
    def getValvePosition(self, valveNumber):
        '''
        Get the state of a valve - 0 to 23
        Returns True or False if valve is set or not - None if error
        '''
        if not self.__connectionState>URBasic.connectionState.ConnectionState.DISCONNECTED:
            self.__logger.warning('GetValvePosition but not Connected')
            return

        #Valves are 0 to 11 - todo make input validation
        result = self.__client.read_coils(valveNumber, 1)
        if(result == None): #Just one retry
            time.sleep(0.2)
            result = self.__client.read_coils(valveNumber, 1)
        if(result != None):
            return result.bits[0]
        else:
            return None
        
        