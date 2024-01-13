#!/usr/bin/env python3
from collections import deque
import rospy
import arcpy
import time
from kident2.msg import Array_f64
from kident2.srv import Get_qResponse, Get_q
import numpy as np
import tf
from parameter_estimator import ParameterEstimator
import utils
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Transform, Vector3, Quaternion
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header, String
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse, Empty, EmptyResponse
import pandas as pd
import sys, select, termios, tty
import cv2
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
from sensor_msgs.msg import Image  # Image is the message type
from robot import RobotDescription


class IiwaHandler:
    """
    Handles an instance of the class "arcpy.Robots.Iiwa"
    Subscribes to the topic "q_desired" and executes the requested trajectories
    Stores past values of q together with time and provides a service that 
    Broadcasts transforms for joints and found markers on ROS tf
    """

    def __init__(self, traj_file) -> None:
        self.netif_addr = '127.0.0.1'  # loopback to gazebo
        # netif_addr = '192.168.1.3'
        try:
            self.iiwa = arcpy.Robots.Iiwa(self.netif_addr)
        except Exception as e:
            rospy.logerr("Instance of Iiwa robot could not be created: {}".format(e))
            pass
        time.sleep(0.1)

        # self.pub_q = rospy.Publisher("iiwa_q", Array_f64, queue_size=20)
        # self.serv_q = rospy.Service('get_q_interp', Get_q, self.get_q_interpolated)
        self.serv_next = rospy.Service('next_move', Trigger, self.move_next_point)
        self.serv_print_state = rospy.Service('print_robot_state', Empty, self.print_robot_state)
        self.pub_status = rospy.Publisher("robot_status", String, queue_size=20)
        self.sub_q_desired = rospy.Subscriber('goto_q_desired', Array_f64, self.move_specific_point)
        self.sub_img = rospy.Subscriber('r1/camera/image', Image, self.broadcast_marker_tfs)
        self.k = 0
        self.forward = True
        try:
            df = pd.read_csv(traj_file)
        except:
            rospy.logerr("Could not load trajectory from {}".format(traj_file))
        self.traj = df.to_numpy()  # shape: (num_joints, num_traj_points)
        self.traj = self.traj[:, 1:]  # delete header

        self.qs = deque(maxlen=10000)
        # self.brs = [tf.TransformBroadcaster() for i in range(9)]
        self.q_desired = np.zeros(7)
        self.state = 'init'
        self.in_motion = False
        self.bridge = CvBridge()
        self.tfbroadcaster = tf.TransformBroadcaster()

    def readout_q(self) -> None:
        """
        Read raw joint data from iiwa.model
        """
        q_raw = self.iiwa.model.get_q()
        q_raw_act = self.iiwa.state.get_q_act()
        t = rospy.get_time()
        self.qs.append((q_raw, t))

    def print_robot_state(self, arg):
        print(self.iiwa.state)
        return EmptyResponse()
    #
    # def get_q_interpolated(self, req) -> Get_qResponse:
    #     """
    #     Implementation of Service "Get_q":
    #     Receives a float time. Find the readout values
    #     before and after requested time. Interpolate for
    #     each q and return interpolated values
    #     """
    #     # initialize response
    #     resp = Get_qResponse()
    #
    #     qs = self.qs
    #     t_interest = req.time
    #
    #     # if t_interest is too long ago return
    #     if t_interest < qs[0][1]:
    #         rospy.logwarn("Cannot provide q for time {}, out of saved area [past]".format(t_interest))
    #         return resp
    #
    #     while t_interest > qs[-1][1]:
    #         pass  # wait for new readout
    #
    #     rospy.sleep(0.01)
    #     for (i, q_elem) in enumerate(qs):  # iterating from oldest entries towards newer ones
    #         q, t = q_elem
    #         if t > t_interest:  # first q_elem where t>t_interest, now t_interest is bw current and prev element
    #             qs_interval = [qs[i - 1][0], qs[i][0]]  # extract q from current and previous element
    #             t_interval = [qs[i - 1][1], qs[i][1]]  # extract t from current and previous element
    #             break
    #             # check if t_interest is newer than last entry
    #
    #     # interpolation
    #     try:
    #         qs_interval = np.transpose(np.array(qs_interval).squeeze())  # convert to array and transpose
    #         q_interp = np.array([np.interp(t_interest, t_interval, qi_interval) for qi_interval in
    #                              qs_interval])  # for each [qi-,qi+] interpolate
    #     except:
    #         rospy.logwarn("iiwa_handler: Could not interpolate")
    #         return resp
    #     resp.q = q_interp.tolist()
    #     resp.time = t_interest
    #     return resp

    # def publish_q(self) -> None:
    #     """
    #     Publish the newest value from stored q readouts
    #     """
    #     msg = Array_f64()
    #     msg.data, msg.time = self.qs[-1]  # publish newest element
    #     self.pub_q.publish(msg)

    def move_next_point(self, arg):
        if not (self.state == 'ready'):
            rospy.loginfo('ROBOT NOT READY')
            res = TriggerResponse()
            res.success = False
            res.message = "ROBOT NOT READY"
            return res
        else:
            self.q_desired = self.traj[:, self.k]
            rospy.loginfo(f'Point {self.k} of traj is {self.q_desired}')
            rospy.loginfo('MOVE NEXT Q DESIRED')
            t0 = self.iiwa.get_time()
            self.iiwa.move_jointspace(self.q_desired, t0, 5, N_pts=10)  # 5 s trajectory
            # iterate forwards and backwards over the array

            if self.forward:
                self.k += 1
            else:
                self.k -= 1
            _, len_traj = np.shape(self.traj)
            if self.k == len_traj - 1:
                self.forward = False
                rospy.loginfo("move_iiwa: REACHED END, GOING BACK")
            if self.k == 0:
                self.forward = True
        res = TriggerResponse()
        res.success = True
        res.message = "NEXT POINT REQUESTED"
        return res

    def move_specific_point(self, msg):
        q = msg.data
        if not (self.state == 'ready'):
            rospy.loginfo('ROBOT NOT READY')
        else:
            self.q_desired = q
            rospy.loginfo(f'Point from topic goto_q_desired {q}')
            rospy.loginfo('MOVE NEXT Q DESIRED')
            t0 = self.iiwa.get_time()
            self.iiwa.move_jointspace(self.q_desired, t0, 5, N_pts=10)  # 5 s trajectory
            # iterate forwards and backwards over the array

    def check_status(self):
        _msg = rospy.wait_for_message("r1/joint_states", JointState)
        speed = np.max(np.abs(np.array(_msg.velocity)))
        epsilon = 0.005

        if speed > epsilon:
            # was stopped but now moving
            if not self.in_motion:
                rospy.loginfo('READY - > BUSY')
            self.state = 'busy'

        if self.state == 'busy':
            if self.in_motion and speed <= epsilon:
                # was moving but now stopped
                self.state = 'ready'
                rospy.loginfo('BUSY - > READY')

        if self.state == 'init':
            if speed <= epsilon:
                self.state = 'ready'
                rospy.loginfo('INIT - > READY')
        self.in_motion = speed > epsilon
        msg = String()
        msg.data = self.state
        self.pub_status.publish(msg)

    def release_udp_socket(self):
        try:
            self.iiwa.udp_socket.close()
        except:
            rospy.logerr("could not close udp socket")

    def broadcast_joint_tfs(self):
        # broadcast the frames for the robot joints
        qs, t = self.qs[-1]
        qs = qs.flatten()
        joint_tfs = RobotDescription.get_joint_tfs(qs)

        for transform in joint_tfs:
            rotmat, trans = utils.split_H_transform(transform['mat'])
            r = R.from_matrix(rotmat)
            quat = r.as_quat()
            self.tfbroadcaster.sendTransform(trans,
                                             quat,
                                             rospy.Time.now(),
                                             'r1/' + transform['from_frame'],
                                             'r1/' + transform['to_frame'])

    def broadcast_marker_tfs(self, data):
        """
        Callback for the camera image subscriber
        gets the estimated pose from camera to marker for each marker found in the image,
        and publishes them on tf
        """
        current_frame = self.bridge.imgmsg_to_cv2(data)
        camera_tfs = RobotDescription.get_camera_tfs(current_frame)

        for transform in camera_tfs:
            rotmat, trans = utils.split_H_transform(transform['mat'])
            r = R.from_matrix(rotmat)
            quat = r.as_quat()
            self.tfbroadcaster.sendTransform(trans,
                                             quat,
                                             rospy.Time.now(),
                                             'r1/' + transform['from_frame'],
                                             'r1/' + transform['to_frame'])

        self.tfbroadcaster.sendTransform(np.array([0, 0, 0]),  # identity translation
                                         np.array([0, 0, 0, 1]),  # identity rotation
                                         rospy.Time.now(),
                                         'r1/cam',
                                         'r1/9')


# Node
if __name__ == "__main__":
    rospy.init_node('iiwa_handler')
    # handler = IiwaHandler(traj_file="/home/armin/catkin_ws/src/kident2/src/traj.csv")
    handler = IiwaHandler(traj_file="/home/armin/catkin_ws/src/kident2/src/single_marker.csv")
    rospy.on_shutdown(handler.release_udp_socket)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        handler.readout_q()
        handler.broadcast_joint_tfs()
        # marker tfs are broadcast by subscribing to the image topic
        handler.check_status()
        rate.sleep()
