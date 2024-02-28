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
#from .. import URPlus #import if any UPplus modules is needed

class RobotConnector(object):
    '''
    Class to hold all connection to the Universal Robot and plus devises  
         
    Input parameters:

    '''


    def __init__(self,robotModel, host, hasForceTorque=False):
        '''
        Constructor see class description for more info.
        '''
        if(False):
            assert isinstance(robotModel, URBasic.robotModel.RobotModel)  ### This line is to get code completion for RobotModel
        self.RobotModel = robotModel
        self.RobotModel.ipAddress = host
        self.RobotModel.hasForceTorqueSensor = hasForceTorque
        self.RealTimeClient = URBasic.realTimeClient.RealTimeClient(robotModel)
        self.DataLog = URBasic.dataLog.DataLog(robotModel)
        self.RTDE = URBasic.rtde.RTDE(robotModel)
        self.DashboardClient = URBasic.dashboard.DashBoard(robotModel)
        self.ForceTourqe = None
        if hasForceTorque:
            self.ForceTourqe = URplus.forceTorqueSensor.ForceTorqueSensor(robotModel)
        
        logger = URBasic.dataLogging.DataLogging()        
        name = logger.AddEventLogging(__name__)        
        self.__logger = logger.__dict__[name]
        self.__logger.info('Init done')


    def close(self):
        self.DataLog.close()
        self.RTDE.close()
        self.RealTimeClient.Disconnect()
        self.DashboardClient.close()
        if self.ForceTourqe is not None:
            self.ForceTourqe.close()
