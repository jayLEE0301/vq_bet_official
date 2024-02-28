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
from ..URPlus.adam6017 import ADAM6017

__author__ = "Martin Huus Bjerge"
__copyright__ = "Copyright 2017, Rope Robotics ApS, Denmark"
__license__ = "MIT License"

import threading
from .. import URBasic

class AdamReader(threading.Thread):
    
    def __init__(self, host, robotModel=None):
        threading.Thread.__init__(self)
        self.__adam6017 = None
        if host is None:
            return
        if robotModel is None:
            self.__robotModel = URBasic.robotModel.RobotModel()
        else:
            self.__robotModel = robotModel
            
        self.__initDataModel()
        self.__counter = 0    
        self.__stop_flag = False
        self.__adam6017 = ADAM6017(host=host)
        self.start()
    
    
    
    def __readSample(self):
        values = self.__adam6017.getAnalogInputs()
        self.__updateDataModel(values)
        #print(values)
        #self.__counter = self.__counter + 1
        #if(self.__counter % 1000 == 0):
        #    print(self.__counter)
    
    def close(self):
        self.__stop_flag = True
        if self.isAlive():
            self.join()
            #self.ser.close()
    
    def connected(self):
        if(self.__adam6017 is not None):
            return self.__adam6017.connected
        else:
            return False
    
    def run(self):
        self.__stop_flag = False
        #print("start")
        while not self.__stop_flag:
            self.__readSample()
    
    def __updateDataModel(self, values):
            self.__robotModel.dataDir['analogInput0'] = values[0]
            self.__robotModel.dataDir['analogInput1'] = values[1]
            self.__robotModel.dataDir['analogInput2'] = values[2]
            self.__robotModel.dataDir['analogInput3'] = values[3]
            self.__robotModel.dataDir['analogInput4'] = values[4]
            self.__robotModel.dataDir['analogInput5'] = values[5]
            self.__robotModel.dataDir['analogInput6'] = values[6]
            self.__robotModel.dataDir['analogInput7'] = values[7]
    
    def __initDataModel(self):
            self.__robotModel.dataDir['analogInput0'] = None
            self.__robotModel.dataDir['analogInput1'] = None
            self.__robotModel.dataDir['analogInput2'] = None
            self.__robotModel.dataDir['analogInput3'] = None
            self.__robotModel.dataDir['analogInput4'] = None
            self.__robotModel.dataDir['analogInput5'] = None
            self.__robotModel.dataDir['analogInput6'] = None
            self.__robotModel.dataDir['analogInput7'] = None

           
    
