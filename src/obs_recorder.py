#!/usr/bin/env python3
import numpy as np
import cv2
from cv_bridge import CvBridge
import rosbag
import pickle
import os
import time
import utils
from parameter_estimator import ParameterEstimator
from robot import RobotDescription
import pandas as pd



###############  SETUP  #####################
# input_bag_file = '/media/armin/Armin/2007_gazebo_iiwa_stopping.bag'
# input_bag_file = '/home/armin/single_marker_2023-11-01-11-12-21.bag'
input_bag_file = '/home/armin/Desktop/rosbags/exp_26_04_002_2024-04-26-12-06-53.bag'
image_topic = 'r1/camera/image'
q_topic = 'r1/joint_states'
#############################################


def process_bag_file(input_bag_file, image_topic, q_topic):
    print('Beginning Processing')
    input_bag = rosbag.Bag(input_bag_file, 'r')  # input file

    bridge = CvBridge()
    observations = []

    # extract q measurements together with timestamps, and store them
    q_timestamps = []
    q_values = []
    for idx, (topic, msg, t) in enumerate(input_bag.read_messages(topics=q_topic)):
        if idx % 100 == 0:
            print(f'finished processing joint values up to {idx}')
        q_timestamps.append(msg.header.stamp.to_sec())
        q_values.append(np.array(msg.position))

    # extract each image together with its timestamp
    for idx, (topic, msg, t) in enumerate(input_bag.read_messages(topics=image_topic)):
        if idx % 100 == 0:
            print(f'finished processing image frames up to {idx}')
        cv_img_timestamp = msg.header.stamp.to_sec()
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        interp_distance = utils.get_interp_distance(q_timestamps, cv_img_timestamp)  # check how far this timestamp is from the closest timestamp for q
        q_interp = utils.interpolate_vector(cv_img_timestamp, q_timestamps, np.array(q_values).T)  # get the interpolated value
        list_obs_frame = RobotDescription.observe(cv_img, q_interp, cv_img_timestamp)  # returns list of obs dictionaries
        _ = [obs.update(interp_dist=interp_distance) for obs in list_obs_frame]  # add interp distance to each dict
        observations.extend(list_obs_frame)  # add all entries of small list to big list

    df = pd.DataFrame(observations)

    return df


recorded_observations = process_bag_file(input_bag_file, image_topic, q_topic)
timestr = time.strftime("%Y%m%d-%H%M%S")
observations_file_str = "obs_{}_{}.p".format(os.path.splitext(os.path.basename(input_bag_file))[0], timestr)

pd.to_pickle(recorded_observations, observations_file_str)

print("file saved as {}".format(observations_file_str))
