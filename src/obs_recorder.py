#!/usr/bin/env python3
import numpy as np
import cv2
from cv_bridge import CvBridge
import rosbag
import pickle
import os
import time


def observe(image, q, aruco_param_dict):
    list_obs = []
    # get marker corners and ids
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, aruco_param_dict['arucoDict'])
    try:
        # pose estimation to return the pose of each marker in the camera frame of reference
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                       aruco_param_dict['aruco_length'],
                                                                       aruco_param_dict['camera_matrix'],
                                                                       aruco_param_dict['camera_distortion'])
    except Exception as e:
        print("Pose estimation failed: {}".format(e))
        return

    if isinstance(ids, type(None)):  # if no markers were found, return
        return {}

    for o in zip(ids, rvecs, tvecs):  # create a dict for each observation
        o_id = o[0][0]
        obs = {"id": o_id,
               "rvec": o[1].flatten().tolist(),
               "tvec": o[2].flatten().tolist(),
               "t": t,
               "q": q}
        list_obs.append(obs)  # append observation to queue corresponding to id (deque from right)

    return list_obs


###############  SETUP  #####################
# input_bag_file = '/media/armin/Armin/2007_gazebo_iiwa_stopping.bag'
input_bag_file = '/home/armin/single_marker_2023-11-01-11-12-21.bag'
image_topic = 'r1/camera/image'
q_topic = 'r1/joint_states'
#############################################

print('begin')

input_bag = rosbag.Bag(input_bag_file, 'r')  # input file

camera_matrix = np.array([1386.4138492513919, 0.0, 960.5, 0.0, 1386.4138492513919, 540.5, 0.0, 0.0, 1.0]).reshape(3, 3)
aruco_params = {'arucoDict': cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000),
                'aruco_length': 0.4,
                'camera_matrix': camera_matrix,
                'camera_distortion': np.zeros(5)}

bridge = CvBridge()
observations = {}
num_obs = 0


# sort q values from rosbag into dictionary
q_values = {'timestamp': [], 'q0': [], 'q1': [], 'q2': [], 'q3': [], 'q4': [], 'q5': [], 'q6': []}
for idx, (topic, msg, t) in enumerate(input_bag.read_messages(topics=q_topic)):
    q_values['timestamp'].append(t.to_sec())
    print(f'reading in joint value {idx}')
    for i in range(7):
        q_values['q{}'.format(i)].append(msg.position[i])

# for each image in the series interpolate the joint coordinates and perform obs
for idx, (topic, msg, t) in enumerate(input_bag.read_messages(topics=image_topic)):
    print('processing frame {}'.format(idx))
    cv_image_timestamp = msg.header.stamp.to_sec()
    cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")

    q_image = []
    for i in range(7):
        qi = np.interp(cv_image_timestamp, q_values['timestamp'], q_values['q{}'.format(i)])
        q_image.append(qi)
    q_image = np.array(q_image)
    list_obs_img = observe(cv2_img, q_image, aruco_params)  # observations made on that frame
    for obs in list_obs_img:  # sort all observations into a dictionary based on tag id
        marker_id = obs['id']
        if not marker_id in observations:  # if this id was not yet used initialize list for it
            observations[marker_id] = []
        observations[marker_id].append(obs)
        num_obs = num_obs + 1
    print(f'num obs: {num_obs}')


timestr = time.strftime("%Y%m%d-%H%M%S")
observations_file_str = "obs_{}_{}.p".format(os.path.splitext(os.path.basename(input_bag_file))[0], timestr)
observations_file = open(observations_file_str, 'wb')
pickle.dump(observations, observations_file)
observations_file.close()
print("file saved as {}".format(observations_file_str))
