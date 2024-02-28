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
import threading
#from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.client.sync import ModbusSerialClient as ModbusClient
import numpy as np


class Mib(threading.Thread):
    '''
    Driver for controlling DEIF MIB 7000C electical multi meter Modbus serial
    
    Example how to use (in my case COM3):
    
    mib = URplus.mib.Mib(host='COM3')
    print(mib.GetVoltage())
    
    Note: If using BarinBoxes ES-313 Ethernet to serial device, 
    remember to remove the tag in port default settings that ignores the application settings.
    http://[IP]/serialport1.html
    '''
    
    
    def __init__(self, host, robotModel=None):
        '''
        Constructor - takes the serial port of the sander
        '''
        threading.Thread.__init__(self)
        logger = URBasic.dataLogging.DataLogging()
        name = logger.AddEventLogging(__name__,log2Consol=False)        
        self.__logger = logger.__dict__[name]
        
        if(False):
            assert isinstance(robotModel, URBasic.robotModel.RobotModel)  ### This line is to get code completion for RobotModel
        if robotModel is None:
            self.__robotModel = URBasic.robotModel.RobotModel()
        else:
            self.__robotModel = robotModel
        self.__initDataModel()
        
        if host is None:
            return

        self.client = ModbusClient(port=host, baudrate='19200', parity='N', stopbits=1, method='RTU')
        self.__serialUnit = 17
        self.__PT1 = 400
        self.__PT2 = 400
        self.__CT1 = 12
        self.__CT2 = 5
        self.__stopRunningFlag = False
        self.__stop_event = True
        self.start()
        self.__logger.info('Mib constructor done')

        
    def close(self):
        self.__stop_event = True
        if self.isAlive():
            self.join()
            self.client.close()

    
    def run(self):
        self.__stop_event = False
        while not self.__stop_event:
            data = np.array(self.client.read_holding_registers(304, 14, unit=self.__serialUnit).registers)
            self.__robotModel.dataDir['mibGridFrequency'] = data[0]/100
            self.__robotModel.dataDir['mibGridVoltge'] = data[1:7]*self.__PT1/self.__PT2/10
            self.__robotModel.dataDir['mibGridCurrent'] = data[7:10]*self.__CT1/self.__CT2/1000 
            self.__robotModel.dataDir['mibGridPower'] = data[11:14]*self.__PT1/self.__PT2*self.__CT1/self.__CT2
            self.__robotModel.dataDir['mibGridEnergy'] = np.array(self.client.read_holding_registers(156, 1, unit=self.__serialUnit).registers)/10
            self.__robotModel.dataDir['mibGridMaxCurrent'] = np.array(self.client.read_holding_registers(1126, 3, unit=self.__serialUnit).registers)*self.__CT1/self.__CT2/1000
    
    def GetFrequency(self):
        return self.__robotModel.dataDir['mibGridFrequency']
    
    def GetVoltage(self):
        return self.__robotModel.dataDir['mibGridVoltge']
    
    def GetCurrent(self):
        return self.__robotModel.dataDir['mibGridCurrent']
    
    def GetPower(self):
        return self.__robotModel.dataDir['mibGridPower']
    
    def GetEnergy(self):
        return self.__robotModel.dataDir['mibGridEnergy']
    
    def GetMaxCurrent(self):
        return self.__robotModel.dataDir['mibGridMaxCurrent']
    
    def __initDataModel(self):
        self.__robotModel.dataDir['mibGridFrequency'] = None
        self.__robotModel.dataDir['mibGridVoltge'] = None
        self.__robotModel.dataDir['mibGridCurrent'] = None
        self.__robotModel.dataDir['mibGridPower'] = None
        self.__robotModel.dataDir['mibGridEnergy'] = None
        self.__robotModel.dataDir['mibGridMaxCurrent'] = None
        