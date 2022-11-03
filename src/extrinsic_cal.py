#!/usr/bin/env python3
import time
import numpy as np
import cv2
import rospy
import ros_numpy
from sensor_msgs.msg import Image, CameraInfo
from kident2.srv import Get_q, Get_qResponse
from kident2.msg import DiffMeasurement
from collections import deque
import random
import utils
from parameter_estimator import ParameterEstimator, RLS
from scipy import linalg
from Pose_Estimation_Class import UKF
import pickle


class ExtrinsicCalib:
    """
    """

    def __init__(self) -> None:
        # data for aruco estimation to be moved to rosparam from constructor

        self.theta_nom = ParameterEstimator.dhparams["theta_nom"]
        self.d_nom = ParameterEstimator.dhparams["d_nom"]
        self.r_nom = ParameterEstimator.dhparams["r_nom"]
        self.alpha_nom = ParameterEstimator.dhparams["alpha_nom"]

        self.M = np.zeros((3, 3))

        rospy.loginfo("Tracker waiting for service get_q_interp")
        rospy.wait_for_service('get_q_interp')
        self.get_q_interp_proxy = rospy.ServiceProxy('get_q_interp', Get_q)
        self.pub_image_overlay = rospy.Publisher("image_overlay", Image, queue_size=20)

        self.observations = dict()  # keys are marker ids, for each id there is a deque of observations

        self.pub_meas = rospy.Publisher("diff_meas", DiffMeasurement, queue_size=20)
        rospy.loginfo("Tracker initialized")

        self.frame_prescaler = 0
        self.C = np.zeros((0, 3))
        self.d = np.empty(0)

        self.ukf = UKF()

        self.cntr = 0
        observations_file_str = "/home/armin/catkin_ws/src/kident2/src/observations.p"
        observations_file = open(observations_file_str, 'rb')

        # dump information to that file
        self.observations = pickle.load(observations_file)
        # close the file
        observations_file.close()



    def process_observations(self):
        """
        Go over each marker tracked in observations and calculate the pose difference between
        the last and second to last observation in world coordinates
        Construct the matrices defining the transformations from camera to marker and from robot base
        to robot endeffector. Use a recursive least squares approach to estimate the transformation
        from endeffector to camera
        """
        try:
            tracked_ids = list(self.observations.keys())  # keys of dict to list
            if (tracked_ids == []):  # dict is empty, nothing to do here
                return
            id = random.choice(tracked_ids)  # pick a random id
            # print("random id: {}".format(id))
            if (len(self.observations[id]) < 2):  # a minimum of 2 observations needed for differential measurement
                return
            obs1 = self.observations[id].popleft()  # this oldest observation is removed from queue
            obs2 = self.observations[id][0]  # this now oldest observation will be kept for next measurement
            timediff = obs2["t"] - obs1["t"]
            # if (np.abs(timediff) > 0.1):  # more than three frames means data is too old, 0.03 s bw frames
                # rospy.logwarn("Measurement dropped, time bw obs was too much with {} s".format(timediff))
                # return
            H1 = utils.H_rvec_tvec(obs1["rvec"], obs1["tvec"])
            H2 = utils.H_rvec_tvec(obs2["rvec"], obs2["tvec"])
            H1inv, H2inv = np.linalg.inv(H1), np.linalg.inv(H2)

            num_links = self.theta_nom.size
            theta_nom1 = np.array(list(obs1["q"])) + self.theta_nom
            T1 = ParameterEstimator.get_T__i0(num_links, theta_nom1, self.d_nom, self.r_nom, self.alpha_nom)

            theta_nom2 = np.array(list(obs2["q"])) + self.theta_nom
            T2 = ParameterEstimator.get_T__i0(num_links, theta_nom2, self.d_nom, self.r_nom, self.alpha_nom)

            AA = np.linalg.inv(T1) @ T2
            # BB = H1 @ H2inv
            # BB = H1inv @ H2
            BB = H2 @ H1inv
            # BB = H2inv @ H1

            self.ukf.Update(AA, BB)
            self.cntr = self.cntr + 1
        except:
            return
        if self.ukf.consistency[-1] > 0.03:
            print("fuggup")
        if self.cntr>100:
            print("ukf.x {}".format(self.cntr))
            print(self.ukf.x)
            print("\n")
            import matplotlib.pyplot as plt
            plt.plot(self.ukf.consistency)
            plt.show()
            print("test")




# Node
if __name__ == "__main__":
    rospy.loginfo('Launching Extrinsic Calibrator')
    calibrator = ExtrinsicCalib()

    while not rospy.is_shutdown():
        calibrator.process_observations()
