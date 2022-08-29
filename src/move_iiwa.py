#!/usr/bin/env python3
import pandas as pd
import rospy
import numpy as np
import itertools
from kident2.msg import Array_f64
from pathlib import Path



class MoveRobot():
    """
    Stores the points of the trajectory and the timestep.
    Each timestep the next point of the trajectory is published
    After reaching an end of the trajectory, the direction is reversed
    """

    def __init__(self, trajectory_csv) -> None:
        try:
            df = pd.read_csv(trajectory_csv)
        except:
            rospy.logerr("Could not load trajectory from {}".format(trajectory_csv))
        self.traj=df.to_numpy() # shape: (num_joints, num_traj_points)
        self.pub =  rospy.Publisher("q_desired",Array_f64,queue_size=2)
        self.k = 0
        self.forward=True


    def send_message(self) -> None:
        # create instance and populate with values
        msg = Array_f64()
        msg.data=self.traj[:,self.k]
        msg.time = rospy.get_time()

        # publish to topic
        self.pub.publish(msg)

        # iterate forwards and backwards over the array
        if self.forward==True:
            self.k += 1
        else:
            self.k -= 1
        _, len_traj = np.shape(self.traj)
        if self.k == len_traj-1:
            self.forward = False
        if self.k == 0:
            self.forward = True

# Node
if __name__ == "__main__":
    rospy.init_node('move_iiwa')
    mover = MoveRobot("/home/armin/catkin_ws/src/kident2/src/traj.csv")
    # rate = rospy.Rate(1) # rate of publishing in Hz
    _msg = rospy.wait_for_message("iiwa_q", Array_f64) # wait for iiwa handler to publish first message
    while not rospy.is_shutdown():
        mover.send_message()
        rospy.sleep(0.3)