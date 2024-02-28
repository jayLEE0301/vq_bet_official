from __future__ import division
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
from ..URBasic.connectionState import ConnectionState
from ..URBasic.dashboard import DashBoard
from ..URBasic.dataLog import DataLog
from ..URBasic.dataLogging import DataLogging
#from ..URBasic.kinematic import *
from ..URBasic.manipulation import *
from ..URBasic.realTimeClient import RealTimeClient
from ..URBasic.robotConnector import RobotConnector
from ..URBasic.robotModel import RobotModel
from ..URBasic.rtde import RTDE
from ..URBasic.urScript import UrScript
from ..URBasic.urScriptExt import UrScriptExt
from ..URBasic.robotiq_gripper import RobotiqGripper