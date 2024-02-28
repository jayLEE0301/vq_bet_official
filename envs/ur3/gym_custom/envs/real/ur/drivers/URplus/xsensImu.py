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
import serial
import struct
import threading
#import numpy as np


class XsensImu(threading.Thread):
    '''
    Driver for reading data from Xsens IMU - MTI-10
    
    Example how to use (in my case COM3):
    
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

        self.ser = serial.Serial()
        self.ser.baudrate = 115200
        self.ser.port = host
        self.ser.bytesize = 8
        self.ser.parity = 'N'
        self.ser.stopbits = 1
        self.ser.timeout = None
        self.ser.xonxoff = 0
        self.ser.rtscts=0
        self.ser.open()
        self.__packetCounter=None
        self.__stop_event = True
        self.start()
        self.__logger.info('XsensImu constructor done')

    def close(self):
        self.__stop_event = True
        if self.isAlive():
            self.join()
            self.ser.close()

    def run(self):
        self.__stop_event = False
        while not self.__stop_event:
            self.__readSample()
        

    def __readSample(self):
        dataLen = self.__getSampleHeader()[3]
        self.__readDataBlock(dataLen)
        self.ser.read(1) #CHECKSUM = 
        # Missing implementation of checksum validation

    def __getSampleHeader(self):
        initdata = self.ser.read(4)
        Preamble = initdata[0]
        BID = initdata[1]
        MID = initdata[2]
        if Preamble!=250 or BID!=255 or MID!=54:
            self.__logger.error('Bad measurement header') 
        dataLen = initdata[3]
        if dataLen == 255:
            dataLen = struct.unpack('>H', self.ser.read(2))
        return [Preamble, BID, MID, dataLen]
    
    def __readDataBlock(self, dataLen):
        data = self.ser.read(dataLen)
        dPnt = 0
        while dPnt<dataLen:
            dHead = self.__getDataHeader(data[dPnt:dPnt+3])
            dPnt += 3
            self.__readDataPack(data[dPnt:dPnt+dHead[1]],dHead[0])
            dPnt += dHead[1]
            
    def __getDataHeader(self,data):
            DataId = struct.unpack('>H', data[0:2])
            DataLen = data[2]
            return [DataId[0], DataLen]
    
    def __readDataPack(self,data, dataId):
        if dataId == 4128: # XDI_PacketCounter          
            PacketCounter = struct.unpack('>H', data)[0]
            if self.__packetCounter is None:
                self.__packetCounter = PacketCounter
            else:
                self.__packetCounter+=1
                if self.__packetCounter!=PacketCounter:
                    self.__logger.warning('Lost data packed')
       
        elif dataId == 4192: # XDI_SampleTimeFine  
            self.__robotModel.dataDir['imuSampleTimeFine'] = struct.unpack('>I', data)[0]
        elif dataId == 57376: #XDI_StatusWord
            self.__robotModel.dataDir['imuStatusWord'] = struct.unpack('>I', data)[0]
        elif dataId == 40976: #XDI_RawAccGyrMagTemp 
            self.__robotModel.dataDir['imuRawAccX'] = struct.unpack('>H', data[0:2])[0]
            self.__robotModel.dataDir['imuRawAccY'] = struct.unpack('>H', data[2:4])[0]
            self.__robotModel.dataDir['imuRawAccZ'] = struct.unpack('>H', data[4:6])[0]
            self.__robotModel.dataDir['imuRawGyrX'] = struct.unpack('>H', data[6:8])[0]
            self.__robotModel.dataDir['imuRawGyrY'] = struct.unpack('>H', data[8:10])[0]
            self.__robotModel.dataDir['imuRawGyrZ'] = struct.unpack('>H', data[10:12])[0]
            self.__robotModel.dataDir['imuRawMagX'] = struct.unpack('>H', data[12:14])[0]
            self.__robotModel.dataDir['imuRawMagY'] = struct.unpack('>H', data[14:16])[0]
            self.__robotModel.dataDir['imuRawMagZ'] = struct.unpack('>H', data[16:18])[0]
            self.__robotModel.dataDir['imuRawTemp'] = struct.unpack('>h', data[18:20])[0]
        elif dataId == 40992: #XDI_RawGyroTemp 
            self.__robotModel.dataDir['imuRawTempGyrX'] = struct.unpack('>h', data[0:2])[0]
            self.__robotModel.dataDir['imuRawTempGyrY'] = struct.unpack('>h', data[2:4])[0]
            self.__robotModel.dataDir['imuRawTempGyrZ'] = struct.unpack('>h', data[4:6])[0]
        elif dataId == 16416: #XDI_Acceleration
            self.__robotModel.dataDir['imuAccX'] = struct.unpack('>f', data[0:4])[0]
            self.__robotModel.dataDir['imuAccY'] = struct.unpack('>f', data[4:8])[0]
            self.__robotModel.dataDir['imuAccZ'] = struct.unpack('>f', data[8:12])[0]
        elif dataId == 16400: #XDI_DeltaV 
            self.__robotModel.dataDir['imuDeltaVX'] = struct.unpack('>f', data[0:4])[0]
            self.__robotModel.dataDir['imuDeltaVY'] = struct.unpack('>f', data[4:8])[0]
            self.__robotModel.dataDir['imuDeltaVZ'] = struct.unpack('>f', data[8:12])[0]
            #print('X:' + str(X) + ' - Y:' + str(Y) + ' - Z:' + str(Z))
        elif dataId == 32800: #XDI_RateOfTurn
            self.__robotModel.dataDir['imuGyrX'] = struct.unpack('>f', data[0:4])[0]
            self.__robotModel.dataDir['imuGyrY'] = struct.unpack('>f', data[4:8])[0]
            self.__robotModel.dataDir['imuGyrZ'] = struct.unpack('>f', data[8:12])[0]
        elif dataId == 32816: #XDI_DeltaQ
            self.__robotModel.dataDir['imuDeltaQ0'] = struct.unpack('>f', data[0:4])[0]
            self.__robotModel.dataDir['imuDeltaQ1'] = struct.unpack('>f', data[4:8])[0]
            self.__robotModel.dataDir['imuDeltaQ2'] = struct.unpack('>f', data[8:12])[0]
            self.__robotModel.dataDir['imuDeltaQ3'] = struct.unpack('>f', data[12:16])[0]
        elif dataId == 49184: #XDI_MagneticField
            self.__robotModel.dataDir['imuMagX'] = struct.unpack('>f', data[0:4])[0]
            self.__robotModel.dataDir['imuMagY'] = struct.unpack('>f', data[4:8])[0]
            self.__robotModel.dataDir['imuMagZ'] = struct.unpack('>f', data[8:12])[0]
        elif dataId == 2064: #XDI_Temperature
            self.__robotModel.dataDir['imuTemp'] = struct.unpack('>f', data)[0]
        else:
            self.__logger.warning('Data block not implemented: ' + hex(dataId))            

    
    def __initDataModel(self):
            self.__robotModel.dataDir['imuSampleTimeFine'] = None
            self.__robotModel.dataDir['imuStatusWord'] = None
            self.__robotModel.dataDir['imuRawAccX'] = None
            self.__robotModel.dataDir['imuRawAccY'] = None
            self.__robotModel.dataDir['imuRawAccZ'] = None
            self.__robotModel.dataDir['imuRawGyrX'] = None
            self.__robotModel.dataDir['imuRawGyrY'] = None
            self.__robotModel.dataDir['imuRawGyrZ'] = None
            self.__robotModel.dataDir['imuRawMagX'] = None
            self.__robotModel.dataDir['imuRawMagY'] = None
            self.__robotModel.dataDir['imuRawMagZ'] = None
            self.__robotModel.dataDir['imuRawTemp'] = None
            self.__robotModel.dataDir['imuRawTempGyrX'] = None
            self.__robotModel.dataDir['imuRawTempGyrY'] = None
            self.__robotModel.dataDir['imuRawTempGyrZ'] = None
            self.__robotModel.dataDir['imuAccX'] = None
            self.__robotModel.dataDir['imuAccY'] = None
            self.__robotModel.dataDir['imuAccZ'] = None
            self.__robotModel.dataDir['imuDeltaVX'] = None
            self.__robotModel.dataDir['imuDeltaVY'] = None
            self.__robotModel.dataDir['imuDeltaVZ'] = None
            self.__robotModel.dataDir['imuGyrX'] = None
            self.__robotModel.dataDir['imuGyrY'] = None
            self.__robotModel.dataDir['imuGyrZ'] = None
            self.__robotModel.dataDir['imuDeltaQ0'] = None
            self.__robotModel.dataDir['imuDeltaQ1'] = None
            self.__robotModel.dataDir['imuDeltaQ2'] = None
            self.__robotModel.dataDir['imuDeltaQ3'] = None
            self.__robotModel.dataDir['imuMagX'] = None
            self.__robotModel.dataDir['imuMagY'] = None
            self.__robotModel.dataDir['imuMagZ'] = None
            self.__robotModel.dataDir['imuTemp'] = None
       