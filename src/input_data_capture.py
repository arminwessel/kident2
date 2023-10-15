#!/usr/bin/env python3
from fileinput import filename
import numpy as np
import cv2
import rospy
import ros_numpy
from sensor_msgs.msg import Image
from kident2.srv import Get_q, Get_qResponse
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge
import pandas as pd
from pathlib import Path


class InputDataCapture:
    """
    Captures a video from an image topic, requests joint coordinates and saves them with
    """

    def __init__(self) -> None:
        self.sub_camera_image = rospy.Subscriber("/r1/camera1/image_raw",
                                                 Image,
                                                 self.image_received)

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = None
        rospy.wait_for_service('get_q_interp')
        self.get_q_interp_proxy = rospy.ServiceProxy('get_q_interp', Get_q)
        self.close_cap = rospy.Service('/close_cap', Trigger, self.trigger_response)
        self.bridge = CvBridge()
        self.t1 = None
        self.t2 = None
        self.framerate = None
        self.cntr = 0
        self.q = None

    def image_received(self, image_message: Image) -> None:
        """
        Callback for image received on topic
        """
        if self.t1 == None:
            t1 = image_message.header.stamp
            self.t1 = t1.to_sec()  # to float seconds
            return
        if self.t2 == None:
            t2 = image_message.header.stamp
            self.t2 = t2.to_sec()  # to float seconds
            return
        if self.framerate == None:
            self.framerate = 1 / (self.t2 - self.t1)
            return
        if self.writer == None:
            self.writer = cv2.VideoWriter(filename="recording.mp4",
                                          fourcc=self.fourcc,
                                          fps=self.framerate,
                                          frameSize=(640, 480))
        self.capture_input_data(image_message)

    def capture_input_data(self, frame):
        if not self.writer.isOpened():
            return
        t = rospy.get_time()
        cv_image = ros_numpy.numpify(frame)  # convert image to np array
        # cv_image = self.bridge.imgmsg_to_cv2(frame) # convert to
        self.writer.write(cv_image)  # Writes the next video frame
        self.cntr += 1
        rospy.loginfo("image written - cntr = {}".format(self.cntr))
        res = self.get_q_interp_proxy(t)  # service call to iiwa_handler to interp q(t)
        q = np.array(res.q)
        if np.any(self.q) == None:
            num_joints = q.size
            self.q = np.zeros((num_joints, 0))
            self.num_joints = num_joints
        self.q = np.hstack((self.q, np.reshape(q, (self.num_joints, 1))))

    def trigger_response(self, request):
        savefile_name = 'traj_captured.csv'
        pd.DataFrame(self.q).to_csv(savefile_name)
        rospy.loginfo("Saved data")
        self.writer.release()
        if not self.writer.isOpened():
            rospy.loginfo("Closed video writer")
            return TriggerResponse(
                success=True,
                message="Closed video writer!")
        else:
            rospy.loginfo("Unable to close video writer")
            return TriggerResponse(
                success=False,
                message="Unable to close video writer!")


# Main function.
if __name__ == "__main__":
    rospy.init_node('input_capture_node')  # init ROS node
    rospy.loginfo('#Node input_capture_node running#')
    while not rospy.get_rostime():  # wait for ros time service
        pass
    rospy.loginfo("Ros Time service is available")

    cp = InputDataCapture()

    rospy.spin()
