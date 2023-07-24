#!/usr/bin/env python3
import numpy as np
import cv2
from cv_bridge import CvBridge
import rosbag
import pickle
import os
import time


#############################################
input_bag_file = '/home/armin/2007_gazebo_iiwa_stopping.bag'
sample_rate = 5  # in Hz
image_topic = '/r1/camera/image'
q_topic = '/r1/joint_states'
status_topic = '/robot_status'
disable_oversampling = True

#############################################

input_bag = rosbag.Bag(input_bag_file, 'r')  # input file


test = input_bag.read_messages(topics=[q_topic, image_topic, status_topic])

#for idx, (topic, msg, t) in enumerate(input_bag.read_messages(topics=[q_topic, image_topic, status_topic])):
#    print(f'{msg}')
for elem in test:
    print(elem)
    pass