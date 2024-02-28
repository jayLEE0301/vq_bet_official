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

import logging
import time
import os
import re
from .. import URBasic
import xml.etree.ElementTree as ET
import ast
from six import with_metaclass

class Singleton(type):
    _instances = {}
    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            self._instances[self] = super(Singleton, self).__call__(*args, **kwargs)
        return self._instances[self]


class DataLogging(with_metaclass(Singleton, object)):
    '''
    A module that add general logging functions to the UR Interface framework.
    '''

    def __init__(self,path=None):
        '''
        Constructor that setup a path where log files will be stored.
        '''
        self.directory = None
        self.logDir = None

        self.__developerTestingFlag = False
        self.__eventLogFileMode = 'w'
        self.__dataLogFileMode = 'w'

        configFilename = URBasic.__file__[0:URBasic.__file__.find('URBasic')] + 'logConfig.xml'
        self.__readConfig(configFileName=configFilename)

        self.GetLogPath(path=path, developerTestingFlag=self.__developerTestingFlag)



        self.fileLogHandler = logging.FileHandler(os.path.join(self.directory, 'UrEvent.log'), mode=self.__eventLogFileMode)
        self.fileLogHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.streamLogHandler = logging.StreamHandler()
        self.streamLogHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.fileDataLogHandler = logging.FileHandler(os.path.join(self.directory, 'UrDataLog.csv'), mode=self.__dataLogFileMode)
        self.writeDataLogHeadder = True



    def __readConfig(self, configFileName):
        tree = ET.parse(configFileName)
        logConfig = tree.getroot()
        developerModeTag = logConfig.find('developerMode')
        self.__developerTestingFlag = ast.literal_eval(developerModeTag.text)

        eventLogConfig = logConfig.find('eventLogConfig')
        eventFileModeTag = eventLogConfig.find('fileMode')
        if (eventFileModeTag.text == "Overwrite"):
            self.__eventLogFileMode = 'w'
        elif (eventFileModeTag.text == "Append"):
            self.__eventLogFileMode = 'a'
        else:
            raise ValueError("Not supported eventLogfile mode: " + eventFileModeTag.text)

        dataLogConfig = logConfig.find('dataLogConfig')
        dataFileModeTag = dataLogConfig.find('fileMode')
        if (dataFileModeTag.text == "Overwrite"):
            self.__dataLogFileMode = 'w'
        elif (dataFileModeTag.text == "Append"):
            self.__dataLogFileMode = 'a'
        else:
            raise ValueError("Not supported dataLogfile mode: " + dataFileModeTag.text)



    def GetLogPath(self,path=None, developerTestingFlag=True):
        '''
        Setup a path where log files will be stored
        Path format .\[path]\YY-mm-dd\HH-MM-SS\
        '''
        if path is None:
            path = URBasic.__file__[0:URBasic.__file__.find('URBasic')] + 'log'
        else:
            path = os.path.join(*(re.split('\\\\|/', path)))
        if path[-1:]=='\\' or path[-1:]=='/':
            path = path[0:-1]
        if self.directory is None:
            self.logDir = path
            if developerTestingFlag:
                self.directory = path
            else:
                self.directory =  os.path.join(path, time.strftime("%Y-%m-%d", time.localtime()), time.strftime("%H-%M-%S", time.localtime()))
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
        return self.directory, self.logDir

    def AddEventLogging(self, name='root', log2file=True, log2Consol=True, level = logging.INFO):
        '''
        Add a new event logger, the event logger can log data to a file and also output the log to the console.

        Input Parameters:
        Name (str): The name of the logger the logger name will get the extension event
        Log2file (bool): Set if the log should be stored in a log file
        Log2Consol (bool): Set if the log should be output to the console

        Return parameter:
        Name (str): The logger name including the extension
        '''
        name = name.replace('__', '').replace('.', '_') + 'Event'
        self.__dict__[name] = logging.getLogger(name)
        if log2file:
            self.__dict__[name].addHandler(self.fileLogHandler)
        if log2Consol:
            self.__dict__[name].addHandler(self.streamLogHandler)
        self.__dict__[name].setLevel(level)
        return name

    def AddDataLogging(self,name='root'):
        '''
        Add a new data logger, the data logger will log data to a csv-file.

        Input Parameters:
        Name (str): The name of the logger the logger name will get the extension Data

        Return parameter:
        Name (str): The logger name including the extension
        '''
        name = name+'Data'
        self.__dict__[name] = logging.getLogger(name)
        self.__dict__[name].addHandler(self.fileDataLogHandler)
        self.__dict__[name].setLevel(logging.INFO)
        if self.writeDataLogHeadder:
            self.__dict__[name].info('Time;ModuleName;Level;Channel;UR_Time;Value1;Value2;Value3;Value4;Value5;Value6')
            self.fileDataLogHandler.setFormatter(logging.Formatter('%(asctime)s;%(name)s;%(levelname)s;%(message)s'))
            self.__dict__[name].addHandler(self.fileDataLogHandler)
            self.writeDataLogHeadder = False
        return name
