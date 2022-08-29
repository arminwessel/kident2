#!/usr/bin/env python3
import rospy
from kident2.srv import Get_q, Get_qRequest




rospy.init_node('tester')
rospy.logwarn("started")
# readout
rospy.wait_for_service('get_q_interp')
get_q_interp_proxy = rospy.ServiceProxy('get_q_interp', Get_q)
rospy.logwarn("service_present")
while not rospy.get_time():
    pass
time = rospy.get_time()
rospy.logwarn("time: {}".format(time))

try:
    resp=get_q_interp_proxy(time)
    print(resp)
except rospy.ServiceException as e:
    print("Service call to get q failed: %s"%e)