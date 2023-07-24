#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Empty, EmptyResponse



class Republisher:
    """
    """

    def __init__(self) -> None:
        self.sub_camera = rospy.Subscriber('/r1/camera/image', Image, self.image_received)
        self.sub_joint = rospy.Subscriber('/r1/joint_states', JointState, self.joint_state_received)
        self.sub_status = rospy.Subscriber('/robot_status', String, self.status_received)
        self.pub_camera = rospy.Publisher('r1/camera_image_static', Image, queue_size=20)
        self.pub_joint = rospy.Publisher('r1/joint_states_static', JointState, queue_size=20)
        self.serv_enable = rospy.Service('republish_enable', Empty, self.enable)
        self.status = None
        self.joint_state_msg = None
        self.enable = True

    def image_received(self, msg):
        if not self.enable:
            return
        if self.status == 'ready':
            self.pub_camera.publish(msg)
            self.pub_joint.publish(self.joint_state_msg)
            self.enable = False

    def status_received(self, msg):
        self.status = msg.data

    def joint_state_received(self, msg):
        self.joint_state_msg = msg

    def enable(self, msg):
        self.enable = True
        return EmptyResponse()

if __name__ == "__main__":
    rospy.init_node('republish_filtered')
    republisher = Republisher()
    rospy.spin()


