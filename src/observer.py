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
from collections import deque
import random
import utils


class Observer:
    """
    Tracks markers in an image feed, estimates their pose and stores these observations
    together with timestamps and corresponding q values in a deque.
    From the other end of the observations are removed in pairs of two. These pairs
    are packaged as a measurement and are published to the topic diff_meas.
    """

    def __init__(self, aruco_marker_length=0.12, cam_matrix=np.eye(3), camera_distortion=np.zeros(5)) -> None:
        # data for aruco estimation to be moved to rosparam from constructor

        self.sub_camera_image = rospy.Subscriber("/r1/camera/image",
                                                 Image,
                                                 self.image_received)
        # self.sub_camera_image = rospy.Subscriber("/image_publisher_1662724020477981875/image_raw",
        #                                          Image,
        #                                          self.image_received)

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)  # All 5by5 Markers
        self.arucoParams = cv2.aruco.DetectorParameters_create()

        self.aruco_length = aruco_marker_length
        self.camera_matrix = cam_matrix
        self.camera_distortion = camera_distortion

        self.locked = False

        self.observations = dict()  # keys are marker ids, for each id there is a deque of observations
        self.num_observations = 0

        self.pub_rvec_tvec = rospy.Publisher("rvec_tvec", Array_f64, queue_size=20)

        rospy.loginfo("Observer Initialized")

    def image_received(self, image_message: Image) -> None:
        """
        Method executed for every frame: get marker observations and joint coordinates
        """
        if self.locked:
            return
        t = image_message.header.stamp.to_sec()
        cv_image = ros_numpy.numpify(image_message)  # convert image to np array
        res = self.get_q_interp_proxy(t)  # service call to iiwa_handler to interp q(t)
        q = np.array(res.q)
        if q.size == 0:
            return  # if interpolation was not successful, dont observe the markers
        self.observe_markers(cv_image, t, q)  # adds observations to queue

    def observe_markers(self, frame, t, q) -> None:
        """
        Method to find markers and estimate their pose in camera frame of reference
        Called for each received frame. Observations are stored as a deque for each marker id
        These queues are stored in the dictionary self.observations
        """
        # get marker corners and ids
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.arucoDict)
        try:
            # pose estimation to return the pose of each marker in the camera frame of reference
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                           self.aruco_length,
                                                                           self.camera_matrix,
                                                                           self.camera_distortion)
        except Exception as e:
            rospy.logwarn("Pose estimation failed: {}".format(e))
            return

        # generate and publish image stream with marker overlays for visualization
        #overlay_frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        #self.pub_image_overlay.publish(ros_numpy.msgify(Image,
        #                                                overlay_frame.astype(np.uint8),
        #                                                encoding='rgb8'))

        if isinstance(ids, type(None)):  # if no markers were found, return
            overlay_frame = cv2.putText(frame, # numpy array on which text is written
                                         "No marker detected", # text
                                         (30, 30), # position at which writing has to start
                                         cv2.FONT_HERSHEY_SIMPLEX, # font family
                                         1, # font size
                                         (255, 0, 0, 255), # font color
                                         3) # font stroke
            self.pub_image_overlay.publish(ros_numpy.msgify(Image,
                                                            overlay_frame.astype(np.uint8),
                                                            encoding='rgb8'))
            return

        for (rvec,tvec) in zip(rvecs,tvecs):
            overlay_frame = cv2.drawFrameAxes(frame, self.camera_matrix, self.camera_distortion, rvec, tvec, self.aruco_length)
        self.pub_image_overlay.publish(ros_numpy.msgify(Image,
                                                        overlay_frame.astype(np.uint8),
                                                        encoding='rgb8'))




        for o in zip(ids, rvecs, tvecs):  # create a dict for each observation
            id = o[0][0]
            obs = {"id": id,
                   "rvec": o[1].flatten().tolist(),
                   "tvec": o[2].flatten().tolist(),
                   "t": t,
                   "q": q}
            self.num_observations += 1 # count observations
            print(f"Number of observations: {self.num_observations}")

            if not id in self.observations:  # if this id was not yet used initialize queue for it
                self.observations[id] = deque(maxlen=50)

            self.observations[id].append(obs)  # append observation to queue corresponding to id (deque from right)