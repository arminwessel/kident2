#!/usr/bin/env python3
import pandas as pd
import rospy
import numpy as np
import itertools
from kident2.msg import Array_f64
from pathlib import Path


class MoveRobot:
    """
    Stores the points of the trajectory and the timestep.
    Each timestep the next point of the trajectory is published
    After reaching an end of the trajectory, the direction is reversed
    """

    def __init__(self, traj_file) -> None:
        self.pub = rospy.Publisher("q_desired", Array_f64, queue_size=2)
        self.k = 0
        self.forward = True
        try:
            df = pd.read_csv(traj_file)
        except:
            rospy.logerr("Could not load trajectory from {}".format(traj_file))
        self.traj = df.to_numpy()  # shape: (num_joints, num_traj_points)
        self.traj = self.traj[:, 1:]  # delete header
        # move certain joints only
        # len_traj =
        # self.traj[0, :] = np.zeros(5000)
        # self.traj[1, :] = np.zeros(5000)
        # self.traj[2, :] = np.zeros(5000)
        # self.traj[3, :] = np.zeros(5000)
        # self.traj[4, :] = np.zeros(5000)
        # self.traj[5, :] = np.zeros(5000)
        # self.traj[6, :] = np.zeros(5000)

    def send_next_q_desired(self):
        q_desired = self.traj[:, self.k]
        # create instance and populate with values
        msg = Array_f64()
        msg.data = q_desired
        msg.time = rospy.get_time() + 5 # 5 seconds to complete the trajectory

        # publish to topic
        self.pub.publish(msg)

        # iterate forwards and backwards over the array
        if self.forward == True:
            self.k += 1
        else:
            self.k -= 1
        _, len_traj = np.shape(self.traj)
        if self.k == len_traj - 1:
            self.forward = False
            rospy.loginfo("move_iiwa: REACHED END, GOING BACK")
        if self.k == 0:
            self.forward = True
        return q_desired


# Node
if __name__ == "__main__":
    rospy.init_node('move_iiwa')
    eps = 10 * np.pi / 180 # precision for norm of difference bw q and q_desired

    mover = MoveRobot(traj_file="/home/armin/catkin_ws/src/kident2/src/traj.csv")
    rate = rospy.Rate(1) # rate in Hz

    _msg = rospy.wait_for_message("iiwa_q", Array_f64) # wait for iiwa handler to publish first message
    rospy.sleep(0.1)
    while not rospy.is_shutdown():
        q_desired = mover.send_next_q_desired()
        while True:
            q_current_msg = rospy.wait_for_message("iiwa_q", Array_f64)
            q_current = q_current_msg.data
            diff = np.array(q_desired)-np.array(q_current)
            test = np.linalg.norm(diff)
            rospy.logerr("diff = {}".format(test))
            if np.linalg.norm(diff) < eps:
                break
            rate.sleep()
