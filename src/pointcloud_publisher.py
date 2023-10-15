#!/usr/bin/env python
import rospy

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs.msg
import pickle
import numpy as np

if __name__ == '__main__':
    '''
    Publishes example pointcloud
    '''
    rospy.init_node('point_publisher')
    pointcloud_publisher = rospy.Publisher("/gazebo_marker_locations", PointCloud)
    pointcloud_publisher2 = rospy.Publisher("/identified_marker_locations", PointCloud)
    rospy.loginfo("pointcloud publisher")
    #giving some time for the publisher to register
    rospy.sleep(0.5)
    #declaring pointcloud
    pointcloud = PointCloud()
    pointcloud2 = PointCloud()
    #filling pointcloud header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'r1/world'
    pointcloud.header = header
    pointcloud2.header = header

    with open('points.p', 'rb') as f:
        pointlist = pickle.load(f)  # deserialize using load()

    with open('marker_locations.p', 'rb') as f:
        pointlist2 = pickle.load(f)  # deserialize using load()

    #filling points
    for point in np.array(pointlist).T:
        pointcloud.points.append(Point32(point[0], point[1], point[2]))

    for point in np.array(pointlist2):
        pointcloud2.points.append(Point32(point[0], point[1], point[2]))

    #publish
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pointcloud_publisher.publish(pointcloud)
        pointcloud_publisher2.publish(pointcloud2)
        rate.sleep()
