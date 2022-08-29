#!/usr/bin/env python3
from collections import deque
import rospy
import arcpy
import time
from kident2.msg import Array_f64
from kident2.srv import Get_qResponse, Get_q
import numpy as np


class IiwaHandler():
    """
    Handles an instance of the class "arcpy.Robots.Iiwa"
    Subscribes to the topic "q_desired" and executes the requested trajectories
    Stores past values of q together with time and provides a service that 
    upon request returns the interpolated value of the joints for a given time
    """

    def __init__(self) -> None:
        self.netif_addr = '127.0.0.1' # loopback to gazebo
        # netif_addr = '192.168.1.3'
        try:
            self.iiwa = arcpy.Robots.Iiwa(self.netif_addr)
        except Exception as e:
            rospy.logerr("Instance of Iiwa robot could not be created: {}".format(e))
            pass
        time.sleep(0.1)

        self.pub_q=rospy.Publisher("iiwa_q", Array_f64, queue_size=20)
        self.serv_q = rospy.Service('get_q_interp', Get_q, self.get_q_interpolated)

        self.qs = deque(maxlen=100)

    def readout_q(self) -> None:
        q_raw = self.iiwa.model.get_q()
        t = rospy.get_time()
        self.qs.append((q_raw,t))

    def get_q_interpolated(self,req):
        # initialize response
        resp = Get_qResponse()

        qs = self.qs
        t_interest = req.time
        
        # if t_interest is too long ago return 
        if t_interest<qs[0][1]:
            rospy.logwarn("Cannot provide q for time {}, out of saved area [past]".format(t_interest))
            return resp
        
        while t_interest>qs[-1][1]:
            pass # wait for new readout
        
        rospy.sleep(0.01)
        for (i,q_elem) in enumerate(qs): # itering from oldest entries towards newer ones
            q,t=q_elem
            if t>t_interest: # first q_elem where t>t_interest, now t_interest is bw current and prev element
                qs_interval = [qs[i-1][0], qs[i][0]] # extract q from current and previous element
                t_interval = [qs[i-1][1], qs[i][1]] # extract t from current and previous element
                break 
                # check if t_interest is newer than last entry


        
        # interpolation
        qs_interval = np.transpose(np.array(qs_interval).squeeze()) # convert to array and transpose
        q_interp = np.array([np.interp(t_interest, t_interval, qi_interval) for qi_interval in qs_interval]) # for each [qi-,qi+] interpolate
        resp.q = q_interp.tolist()
        resp.time = t_interest
        return resp


    def publish_q(self):
        msg = Array_f64()
        msg.data, msg.time = self.qs[-1] # publish newest element
        self.pub_q.publish(msg)





# Node
if __name__ == "__main__":
    rospy.init_node('iiwa_handler')
    handler = IiwaHandler()

    rate = rospy.Rate(10) # rate of readout
    while not rospy.is_shutdown():
        # slot 1/4
        handler.readout_q()
        rate.sleep()

        # slot 2/4
        handler.readout_q()
        rate.sleep()

        # slot 3/4
        handler.readout_q()
        rate.sleep()

        # slot 4/4
        handler.readout_q()
        handler.publish_q() # publish q on every fourth passing
        rate.sleep()
