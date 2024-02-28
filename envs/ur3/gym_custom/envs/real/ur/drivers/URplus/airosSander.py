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


class AirosSander(object):
    '''
    Driver for controlling Mirka Airos Sander via Modbus over to TCP
    
    Example how to use (in my case COM4:
    
    sander = AirosSander(host='COM4')
    sander.powerOn()
    time.sleep(1)
    sander.runSander()
    time.sleep(2)    
    sander.setDesiredSpeed(5000)
    time.sleep(2)
    sander.setDesiredSpeed(8000)
    time.sleep(2)
    sander.stopSander()
    sander.powerOff()
    '''
    
    
    def __init__(self, host):
        '''
        Constructor - takes the serial port of the sander
        '''
        logger = URBasic.dataLogging.DataLogging()
        name = logger.AddEventLogging(__name__,log2Consol=False)        
        self.__logger = logger.__dict__[name]
        self.__client = ModbusClient(port=host, baudrate='19200', parity='E', stopbits=1, method='RTU')
        self.__serialUnit = 86
        self.__whatchdogTime = 300  #300 seconds default
        self.__whatchdog = threading.Timer(interval=self.__whatchdogTime, function=self.__whatchdogExpired)
        self.__stopRunningFlag = False
        
    def close(self):
        self.__stopRunningFlag = True
        self.__whatchdog.cancel()
        self.__client.close()

    def powerOn(self):
        '''
        Power on the sander before setting values and running
        '''
        self.__client.write_register(11, 4, unit=self.__serialUnit)
        self.__logger.info("Sander powered on")
        
    def powerOff(self):
        '''
        Power down sander nicely
        '''
        self.__whatchdog.cancel()
        self.__client.write_register(11, 8, unit=self.__serialUnit)
        self.__logger.info("Sander powered off")
        self.__whatchdog.cancel()
        self.__whatchdog = threading.Timer(interval=self.__whatchdogTime, function=self.__whatchdogExpired)
    
    def runSander(self, rpm=4000):
        '''
        Start the sanding - set anywhere from 4000 to 10000 RPM - default is 4000 RPM 
        call powerOn before running
        '''
        if(self.__stopRunningFlag == False):
            self.setDesiredSpeed(rpm)
            self.__client.write_register(11, 1, unit=self.__serialUnit)
            self.__logger.info("Sander started")
            self.__whatchdog.cancel()
            self.__whatchdog = threading.Timer(interval=self.__whatchdogTime, function=self.__whatchdogExpired)
            self.__whatchdog.start()
        else:
            self.__logger.error("Sander can not be started - has ")
    
    def stopSander(self):
        '''
        Stop the sander from running
        '''
        self.__whatchdog.cancel()
        self.__client.write_register(11, 2, unit=self.__serialUnit)
        self.__logger.info("Sander stopped")
        self.__whatchdog = threading.Timer(interval=self.__whatchdogTime, function=self.__whatchdogExpired)
    
    def setDesiredSpeed(self, rpm):
        '''
        Set the speed in RPM to run the sander at - 4000 to 10000 is allowed
        '''
        if(rpm < 4000 or rpm > 10000):
            raise ValueError("Sander RPM out of range")
        self.__client.write_register(10, rpm, unit=self.__serialUnit)
        self.__logger.info("Sander speed set to " + str(rpm))
        self.__resetWhatchdog()
        
    def __resetWhatchdog(self):
        wasRunning=False
        if(self.__whatchdog.is_alive()):
            wasRunning = True
        self.__whatchdog.cancel()
        self.__whatchdog = threading.Timer(interval=self.__whatchdogTime, function=self.__whatchdogExpired)
        if(wasRunning):
            self.__whatchdog.start()
    
    
    def __whatchdogExpired(self):
        self.__stopRunningFlag = True
        self.__logger.error("Sander ran too long")
        self.stopSander()
        
    
    def getMotorTemperature(self):
        '''
        Note implemented due to Mirka adressing problems
        '''
        #self.__client.read_holding_registers(address, count)
        return self.__client.read_input_registers(18, 1,unit=self.__serialUnit).registers[0]
        
        
    def getPcbTemperature(self):
        '''
        Note implemented due to Mirka adressing problems
        '''
        return self.__client.read_input_registers(19, 1,unit=self.__serialUnit).registers[0]
    