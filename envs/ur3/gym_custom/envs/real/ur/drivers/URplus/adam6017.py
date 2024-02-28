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


from pymodbus.client.sync import ModbusTcpClient as ModbusClient
#from pymodbus.client.sync import ModbusSerialClient as ModbusClient
#from pymodbus.client.sync import ModbusUdpClient as ModbusClient
#from bitstring import Bits

class InputRange:
    PlusMinus150mV = 259
    PlusMinus500mV = 260
    ZeroTo150mV = 261
    ZeroTo500mV = 262
    PlusMinus1V = 320
    PlusMinus5V = 322
    PlusMinus10V = 323
    ZeroTo1V = 325
    ZeroTo5V = 327
    ZeroTo10V = 328
    PlusMinus20mA = 385
    FourTo20mA = 384
    ZeroTo20mA = 386
    
    

class ADAM6017(object):
    '''
    Controlling the additional inputs and outputs on the robot
    This class is setup to use the ADAM-6017 model via Modbus over TCP
    '''
    def __init__(self, host):
        '''
        Constructor takes the ip address off the ADAM6017 box
        '''
        if host is None: #Only for enable code completion
            self.connected = False
            return 
        self.__client = ModbusClient(host=host)
        self.__inputRanges = self.__getAllInputRanges()
        self.connected = self.__client.connect()
        
    def setDigitalOutput(self, outputNumber, state):
        '''
        Set digital output True or False - two output 0 and 1 available
        '''
        result = self.__client.write_coil(outputNumber+16, state)
        self.__inputRanges = self.__getAllInputRanges()
    
    def getDigitalOutputState(self, outputNumber):
        '''
        Get the state of digital output (0 and 1 available)
        Returns True or False - or None if error
        '''
        result = self.__client.read_coils(outputNumber+16,1)
        #result2 = self.__client.read_holding_registers(inputNumber, 2)
        return result.bits[0]
    
    def getAnalogInputs(self):
        '''
        Get the raw value of all the analog inputs at once in raw
        registers and values
        '''
        #Analog inputs return mA and V - only tested on 4-20 mA range
        #counter = 0 
        result = self.__client.read_holding_registers(0, 8)
        for x in range(0,8):
            result.registers[x] = self.__getValueFromReading(reading=result.registers[x], inputRange=self.__inputRanges[x])
            #counter = counter+1
        return result.registers
    
    def getAnalogInput(self, inputNumber):
        '''
        Get the value of an analog input by input number 0 to 7
        Returns a value in mA or V depening on InputRange set on port
        note - currently only teset with range 4-20mA
        '''
        result = self.__client.read_holding_registers(inputNumber, 1)
        return self.__getValueFromReading(reading=result.registers[0], inputRange=self.__inputRanges[inputNumber])
        
    def getInputRange(self, inputNumber):
        '''
        Get the input range  of an analog input by input number 0 to 7 configured in
        the device. 
        Returns a value in mA or V depening on InputRange set on port
        '''
        result = self.__client.read_holding_registers(200+inputNumber, 1)
        return result
    
    def setInputRange(self, inputNumber, inputRange):
        '''
        Set the input range  of an analog input by input number 0 to 7. InputRange
        is defined in InputRange Class
        '''
        result = self.__client.write_register(200+inputNumber, inputRange)
        
    def __getAllInputRanges(self):
        result = self.__client.read_holding_registers(200, 8)
        return result.registers
    
    def __getValueFromReading(self, reading, inputRange):
        if(inputRange == InputRange.FourTo20mA):
            if(reading == 0 or reading> 65500):
                return None     #Reading out of range
            value = 16/65535*reading
            return value+4
        if(inputRange == inputRange.ZeroTo20mA):
            if(reading > 65500):
                return None     #Reading out of range
            return 20/65535*reading
        if(inputRange == InputRange.ZeroTo10V):
            if(reading>65500):
                return None     #Reading out of range
            return 10/65535*reading
        else:
            return None
        
