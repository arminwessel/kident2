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


class MarkerBasedTracker:
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

        rospy.loginfo("Tracker waiting for service get_q_interp")
        rospy.wait_for_service('get_q_interp')
        self.get_q_interp_proxy = rospy.ServiceProxy('get_q_interp', Get_q)
        self.pub_image_overlay = rospy.Publisher("image_overlay", Image, queue_size=20)

        self.observations = dict()  # keys are marker ids, for each id there is a deque of observations
        self.num_observations = 0

        self.pub_meas = rospy.Publisher("diff_meas", DiffMeasurement, queue_size=20)
        rospy.loginfo("Tracker initialized")

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
                self.observations[id] = deque(maxlen=30)

            self.observations[id].append(obs)  # append observation to queue corresponding to id (deque from right)

    def process_observations(self):
        """
        Go over each marker tracked in observations and calculate the pose difference between
        the last and second to last observation in world coordinates
        A differential measurement contains the calculated pose difference, 
        the q values for both observations, and the type "marker" for marker based measurement
        The method is scheduled to run in an infinite loop        
        """
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
        if (np.abs(timediff) > 0.1):  # more than three frames means data is too old, 0.03 s bw frames
            rospy.logwarn("Measurement dropped, time bw obs was too much with {} s".format(timediff))
            return
        H1 = utils.H(obs1["rvec"], obs1["tvec"])
        H2 = utils.H(obs2["rvec"], obs2["tvec"])
        H1i, H2i = np.linalg.inv(H1), np.linalg.inv(H2)
        test = H1i[0:3, 3]
        test2 = H2i[0:3, 3]
        dtvec = np.reshape(H1i[0:3, 3] - H2i[0:3, 3], (3, 1))
        drvec = cv2.Rodrigues(H1i[0:3, 0:3])[0] - cv2.Rodrigues(H2i[0:3, 0:3])[0]

        #if np.linalg.norm(np.vstack((dtvec, drvec))) == 0:  # if no movement return
            #return

        # sanity checks
        dist_pos = np.linalg.norm(dtvec)
        if dist_pos > 0.05:
            rospy.logwarn("Distance in pose is greater than 5 cm: dist={} cm".format(100 * dist_pos))
            return

        dist_q = np.array(obs1["q"]) - np.array(obs2["q"])
        if np.any(dist_q > 0.01):
            rospy.logwarn("Distance q greater than 0.01 rad: dist={} rad".format(dist_q))

        m = DiffMeasurement()
        m.type = "marker"  # measurement based on marker as opposed to laser tracker
        m.dtvec = dtvec.flatten().tolist()
        m.drvec = drvec.flatten().tolist()
        m.q1 = list(obs1["q"])
        m.q2 = list(obs2["q"])

        self.pub_meas.publish(m)


# Node
if __name__ == "__main__":
    rospy.init_node('marker_based_tracker')
    rospy.loginfo('Waiting for camera info')
    data = rospy.wait_for_message('/r1/camera/camera_info', CameraInfo)
    camera_matrix = np.array(data.K)
    camera_matrix = np.reshape(camera_matrix, (3, 3))
    rospy.loginfo('Launching Marker-Based Tracker')
    tracker = MarkerBasedTracker(aruco_marker_length=0.12, cam_matrix=camera_matrix, camera_distortion=np.zeros(5))

    while not rospy.is_shutdown():
        # if tracker.num_observations > 1000:
        num_markers = len(tracker.observations.keys())
        print(f"Num markers: {num_markers}")
        # if tracker.num_observations > 50:
        if num_markers > 80:
            tracker.locked = True
            import pickle
            # open a file to store data
            observations_file_str = "/home/armin/catkin_ws/src/kident2/src/observations.p"
            observations_file = open(observations_file_str, 'wb')
            # dump information to that file
            pickle.dump(tracker.observations, observations_file)
            # close the file
            observations_file.close()
            print("done")
            break
        else:
            rospy.sleep(0.5)
        #tracker.process_observations()
