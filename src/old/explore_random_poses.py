#!/usr/bin/env python3
import time
import numpy as np
import cv2
import rospy
import ros_numpy
from sensor_msgs.msg import Image, CameraInfo
from kident2.msg import Array_f64
from kident2.srv import Get_q, Get_qResponse
from kident2.msg import DiffMeasurement
from std_msgs.msg import String
from collections import deque
import random
import utils
import datetime


class Explorer:
    """
    """

    def __init__(self, aruco_marker_length=0.12, cam_matrix=np.eye(3), camera_distortion=np.zeros(5)) -> None:
        # data for aruco estimation to be moved to rosparam from constructor

        self.sub_camera_image = rospy.Subscriber("/r1/camera/image",
                                                 Image,
                                                 self.image_received)

        self.pub_q_des = rospy.Publisher("goto_q_desired", Array_f64, queue_size=20)
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)  # All 5by5 Markers
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.image = None
        self.image_t = None
        self.aruco_length = aruco_marker_length
        self.camera_matrix = cam_matrix
        self.camera_distortion = camera_distortion
        self.database = []
        rospy.loginfo("Explorer Initialized")
        self.j_lims = {}
        self.j_lims[1] = {"lower": -170, "upper": 170, "effort": 320, "vel": 1.4835}
        self.j_lims[2] = {"lower": -120, "upper": 120, "effort": 320, "vel": 1.4835}
        self.j_lims[3] = {"lower": -170, "upper": 170, "effort": 176, "vel": 1.7453}
        self.j_lims[4] = {"lower": -120, "upper": 120, "effort": 176, "vel": 1.3090}
        self.j_lims[5] = {"lower": -170, "upper": 170, "effort": 110, "vel": 2.2689}
        self.j_lims[6] = {"lower": -120, "upper": 120, "effort": 40, "vel": 2.3562}
        self.j_lims[7] = {"lower": -175, "upper": 175, "effort": 40, "vel": 2.3562}

    def image_received(self, image_message: Image) -> None:
        """
        """
        self.image_t = image_message.header.stamp.to_sec()
        cv_image = ros_numpy.numpify(image_message)  # convert image to np array
        self.image = cv_image

    def evaluate_pose(self):
        (corners, ids, rejected) = cv2.aruco.detectMarkers(self.image, self.arucoDict)
        if len(ids) > 0:
            timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            cv2.imwrite(f'marker_{str(ids)}'+timestamp+'.png', self.image)
            res = self.get_q_interp_proxy(self.image_t)  # service call to iiwa_handler to interp q(t)
            q = np.array(res.q)
            self.database.append({'q': q, 'ids': ids})

    def get_random_pose(self):
        q_raw = []
        for i in range(7):
            rand_degree = random.randrange(self.j_lims[i+1]['lower']*100, self.j_lims[i+1]['upper']*100)/100
            q_raw[i] = rand_degree/180*np.pi
        return q_raw

    def goto_random_pose(self):
        pose = self.get_random_pose()
        msg = Array_f64
        msg.data = pose
        msg.time = 0
        self.pub_q_des.publish(msg)

if __name__ == "__main__":
    rospy.init_node('pose explorer')
    explorer = Explorer()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        explorer.goto_random_pose()  # make up random joint variables and send out the message
        rospy.sleep(rate)  # leave some time for status to get busy
        robot_status_msg = String()
        robot_status_msg = rospy.wait_for_message("robot_status")
        if robot_status_msg.data == 'ready':
            explorer.evaluate_pose()  # check if there's a marker visible, then save the pose
            print(explorer.database)

