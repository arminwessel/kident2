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
from std_msgs.msg import Header
from std_srvs.srv import Empty, EmptyResponse
import pandas as pd
import sys, select, termios, tty


class IiwaHandler:
    """
    Handles an instance of the class "arcpy.Robots.Iiwa"
    Subscribes to the topic "q_desired" and executes the requested trajectories
    Stores past values of q together with time and provides a service that 
    upon request returns the interpolated value of the joints for a given time
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

        self.pub_q = rospy.Publisher("iiwa_q", Array_f64, queue_size=20)
        self.serv_q = rospy.Service('get_q_interp', Get_q, self.get_q_interpolated)
        self.serv_next = rospy.Service('next_move', Empty, self.move_next_point)
        self.serv_print_state = rospy.Service('print_robot_state', Empty, self.print_robot_state)
        self.k = 0
        self.forward = True
        try:
            df = pd.read_csv(traj_file)
        except:
            rospy.logerr("Could not load trajectory from {}".format(traj_file))
        self.traj = df.to_numpy()  # shape: (num_joints, num_traj_points)
        self.traj = self.traj[:, 1:]  # delete header

        self.qs = deque(maxlen=10000)
        self.brs = [tf.TransformBroadcaster() for i in range(8)]
        self.q_desired = np.zeros(7)
        self.state = 'init'
        self.in_motion = False

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

    def get_q_interpolated(self, req) -> Get_qResponse:
        """
        Implementation of Service "Get_q":
        Receives a float time. Find the readout values 
        before and after requested time. Interpolate for 
        each q and return interpolated values 
        """
        # initialize response
        resp = Get_qResponse()

        qs = self.qs
        t_interest = req.time
        
        # if t_interest is too long ago return 
        if t_interest < qs[0][1]:
            rospy.logwarn("Cannot provide q for time {}, out of saved area [past]".format(t_interest))
            return resp
        
        while t_interest > qs[-1][1]:
            pass  # wait for new readout
        
        rospy.sleep(0.01)
        for (i, q_elem) in enumerate(qs):  # iterating from oldest entries towards newer ones
            q, t = q_elem
            if t > t_interest:  # first q_elem where t>t_interest, now t_interest is bw current and prev element
                qs_interval = [qs[i-1][0], qs[i][0]]  # extract q from current and previous element
                t_interval = [qs[i-1][1], qs[i][1]]  # extract t from current and previous element
                break 
                # check if t_interest is newer than last entry

        # interpolation
        try:
            qs_interval = np.transpose(np.array(qs_interval).squeeze())  # convert to array and transpose
            q_interp = np.array([np.interp(t_interest, t_interval, qi_interval) for qi_interval in qs_interval])  # for each [qi-,qi+] interpolate
        except:
            rospy.logwarn("iiwa_handler: Could not interpolate")
            return resp
        resp.q = q_interp.tolist()
        resp.time = t_interest
        return resp

    def publish_q(self) -> None:
        """
        Publish the newest value from stored q readouts
        """
        msg = Array_f64()
        msg.data, msg.time = self.qs[-1]  # publish newest element
        self.pub_q.publish(msg)

    def move_next_point(self, arg):
        if not (self.state == 'ready'):
            rospy.loginfo('ROBOT NOT READY')
            return
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
        return EmptyResponse()

    def check_status(self):
        q_dot_set = self.iiwa.state.get_q_dot_set()
        speed = np.sum(np.abs(q_dot_set))

        if speed > 0:
            # was stopped but now moving
            if not self.in_motion:
                rospy.loginfo('READY - > BUSY')
            self.state = 'busy'

        if self.state == 'busy':
            if self.in_motion and speed == 0:
                # was moving but now stopped
                self.state = 'ready'
                rospy.loginfo('BUSY - > READY')

        if self.state == 'init':
            if speed == 0:
                self.state = 'ready'
                rospy.loginfo('INIT - > READY')
        self.in_motion = np.sum(q_dot_set) != 0

    

    def release_udp_socket(self):
        try:
            self.iiwa.udp_socket.close()
        except:
            rospy.logerr("could not close udp socket")

    def broadcast_tf(self):
        names = [f'r1/dh_link_{i}' for i in range(15)]
        theta_nom = ParameterEstimator.dhparams["theta_nom"]
        d_nom = ParameterEstimator.dhparams["d_nom"]
        r_nom = ParameterEstimator.dhparams["r_nom"]
        alpha_nom = ParameterEstimator.dhparams["alpha_nom"]
        qs, t = self.qs[-1]
        qs = qs.flatten()
        qs = np.append(qs, np.zeros(1))
        for (i, q) in enumerate(qs):  # iterate over latest set of joint values
            theta = theta_nom[i]
            d = d_nom[i]
            r = r_nom[i]
            alpha = alpha_nom[i]
            T = ParameterEstimator.get_T_i_forward(q, theta, d, r, alpha)
            quat = tf.transformations.quaternion_from_matrix(T)
            translation = T[0:3, 3]
            # ret = self.brs[i].sendTransform(trans,
            #                       quat,
            #                       rospy.Time.now(),
            #                       names[i+1],
            #                       names[i])
            trans = Transform(translation=Vector3(*translation.tolist()),
                              rotation=Quaternion(*quat.tolist())
                              )

            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = names[i]  # the parent link
            # I use the stamped msg signature call to prevent the order confusion
            trans_stamp = TransformStamped(header, names[i+1], trans)
            self.brs[i].sendTransformMessage(trans_stamp)

        T_corr = np.array([[0, 0, 1, 0],
                           [-1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, 0, 1]])  # euler [ x: -np.pi/2, y: np.pi/2, z: 0 ]

        T_W0 = np.array([[-1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 1, 0.36],
                         [0, 0, 0, 1]])
        quat_W0 = tf.transformations.quaternion_from_matrix(T_W0)
        translation_W0 = T_W0[0:3, 3]
        self.brs[i].sendTransform(translation_W0,
                              quat_W0,
                              rospy.Time.now(),
                              'r1/dh_link_0',
                              'r1/world')

# Node
if __name__ == "__main__":
    rospy.init_node('iiwa_handler')
    handler = IiwaHandler(traj_file="/home/armin/catkin_ws/src/kident2/src/traj.csv")
    rospy.on_shutdown(handler.release_udp_socket)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        handler.readout_q()
        handler.publish_q()  # publish q on every fourth passing
        handler.broadcast_tf()
        handler.check_status()
        rate.sleep()


