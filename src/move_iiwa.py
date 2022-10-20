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

    def __init__(self, mode, traj=None) -> None:
        self.pub = rospy.Publisher("q_desired", Array_f64, queue_size=2)
        self.k = 0
        self.forward = True
        if mode == "csv":
            try:
                df = pd.read_csv(traj)
            except:
                rospy.logerr("Could not load trajectory from {}".format(traj))
            self.traj = df.to_numpy()  # shape: (num_joints, num_traj_points)
            self.traj = self.traj[:, 1:]  # delete header
            # move certain joints only
            self.traj[0, :] = np.zeros(5000)
            self.traj[1, :] = np.zeros(5000)
            self.traj[2, :] = np.zeros(5000)
            self.traj[3, :] = np.zeros(5000)
            self.traj[4, :] = np.zeros(5000)
            # self.traj[5, :] = np.zeros(5000)
            # self.traj[6, :] = np.zeros(5000)
        elif mode == "tri":
            a = 10 / 180 * np.pi
            self.traj = np.array([[0, a], [0, a], [0, a], [0, a], [0, a], [0, a], [0, a]])

    def move_random(self):
        q = (np.random.rand(7) - np.full((7,), 0.5)) * np.pi / 2
        # create instance and populate
        msg = Array_f64()
        msg.data = q
        msg.time = rospy.get_time()

        # publish to topic
        self.pub.publish(msg)

    def send_message(self) -> None:
        # create instance and populate with values
        msg = Array_f64()
        msg.data = self.traj[:, self.k]
        msg.time = rospy.get_time()

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
            rospy.logerr("move_iiwa: REACHED END, GOING BACK")
        if self.k == 0:
            self.forward = True


# Node
if __name__ == "__main__":
    rospy.init_node('move_iiwa')

    mover = MoveRobot(mode='csv', traj="/home/armin/catkin_ws/src/kident2/src/traj.csv")
    # rate = rospy.Rate(1) # rate of publishing in Hz

    # mover = MoveRobot(mode='tri')
    # _msg = rospy.wait_for_message("iiwa_q", Array_f64) # wait for iiwa handler to publish first message
    rospy.sleep(0.1)
    while not rospy.is_shutdown():
        mover.move_random()
        rospy.sleep(1)
