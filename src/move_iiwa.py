#!/usr/bin/env python3
import pandas as pd
import rospy
import numpy as np
import itertools
from kident2.msg import Array_f64
from std_srvs.srv import Empty, EmptyRequest
from pathlib import Path


class MoveRobot:
    """
    Stores the points of the trajectory and the timestep.
    Each timestep the next point of the trajectory is published
    After reaching an end of the trajectory, the direction is reversed
    """

    def __init__(self, traj_file) -> None:
        rospy.wait_for_service('next_move')
        self.request_next_move = rospy.ServiceProxy('next_move', Empty)


# Node
if __name__ == "__main__":
    rospy.init_node('move_iiwa')
    mover = MoveRobot()
    rate = rospy.Rate(0.3)  # rate in Hz
    request = EmptyRequest()
    while not rospy.is_shutdown():
        mover.request_next_move(request)
        rate.sleep()
