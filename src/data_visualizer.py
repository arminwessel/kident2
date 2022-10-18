#!/usr/bin/env python3

from inspect import trace
import rospy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kident2.msg import Array_f64
import numpy as np
from sensor_msgs.msg import Image
import ros_numpy
import time
from pathlib import Path
import pandas as pd
import scipy.signal as sig


class DataVisualizer:
    """
    Store and visualize data
    """
    def __init__(self, num_links) -> None:
        self.sub_est = rospy.Subscriber("est", Array_f64, self.update_est_data)
        self.sub_traj = rospy.Subscriber("iiwa_q", Array_f64, self.update_traj_data)
        self.sub_traj_des = rospy.Subscriber("q_desired", Array_f64, self.update_traj_des_data)

        self.num_links = num_links
        self.num_params = 4*num_links

        self.est_vals = np.empty((num_links*4+6, 0))
        self.est_t = np.empty((0,))
        self.traj_vals = np.empty((num_links, 0))
        self.traj_t = np.empty((0,))
        self.traj_des_vals = np.empty((num_links, 0))
        self.traj_des_t = np.empty((0,))

        self.fig_est, self.ax_est = plt.subplots(2, 2)
        self.fig_est.set_size_inches(16, 9, forward=True)
        self.fig_est.tight_layout(pad=2)

        self.fig_traj, self.ax_traj = plt.subplots(1, 1)
        self.fig_curr_est, self.ax_curr_est = plt.subplots(2, 2)

        # self.pub_plot_est = rospy.Publisher("plot_estimated_params", Image, queue_size=20)
        # self.pub_plot_traj = rospy.Publisher("plot_robot_trajectory", Image, queue_size=20)

        ani_est = animation.FuncAnimation(self.fig_est, self.plot_est, interval=20)
        ani_traj = animation.FuncAnimation(self.fig_traj, self.plot_traj, interval=20)
        ani_curr_est = animation.FuncAnimation(self.fig_curr_est, self.plot_curr_est, interval=20)
        plt.show()

    def update_est_data(self, msg):
        estimate_k = msg.data
        estimate_k = np.reshape(np.array(estimate_k), (-1, 1))
        self.est_vals = np.hstack((self.est_vals, estimate_k))
        self.est_t = np.append(self.est_t, msg.time)

    def update_traj_data(self, msg):
        traj_k = msg.data
        traj_k = np.reshape(np.array(traj_k), (-1, 1))
        self.traj_t = np.append(self.traj_t, msg.time)
        self.traj_vals = np.hstack((self.traj_vals, traj_k))

    def update_traj_des_data(self, msg):
        traj_k = msg.data
        traj_k = np.reshape(np.array(traj_k), (-1, 1))
        self.traj_des_t = np.append(self.traj_des_t, msg.time)
        self.traj_des_vals = np.hstack((self.traj_des_vals, traj_k))

    def plot_est(self, i=None):
        n = self.num_links
        n_tot = self.num_params
        X = self.est_t

        colors = np.array(['tab:blue', 'tab:orange', 'tab:green',
                           'tab:red',  'tab:purple', 'tab:olive',
                           'tab:cyan', 'tab:pink',   'tab:brown', 'tab:gray'])
        if n > colors.size:
            colors = np.random.choice(colors, size=(n,), replace=True, p=None)

        axis = self.ax_est

        param_errors = self.est_vals
        axis[0, 0].clear()
        for i in range(n):
            axis[0, 0].plot(X, param_errors[i, :].flatten(), color=colors[i],   label=str(i))
        axis[0, 0].set_title(r'$\Delta$$\theta$')
        axis[0, 0].legend()

        axis[0, 1].clear()
        for i in range(n):
            axis[0, 1].plot(X, param_errors[i+n, :].flatten(), color=colors[i],   label=str(i))
        axis[0, 1].set_title(r'$\Delta$d')
        axis[0, 1].legend()

        axis[1, 0].clear()
        for i in range(n):
            axis[1, 0].plot(X, param_errors[i+2*n, :].flatten(), color=colors[i],   label=str(i))
        axis[1, 0].set_title(r'$\Delta$a')
        axis[1, 0].legend()

        axis[1, 1].clear()
        for i in range(n):
            axis[1, 1].plot(X, param_errors[i+3*n, :].flatten(), color=colors[i],   label=str(i))
        axis[1, 1].set_title(r'$\Delta$$\alpha$')
        axis[1, 1].legend()

    def plot_traj(self, i=None):
        n=self.num_links
        colors = np.array(['tab:blue', 'tab:orange', 'tab:green',
                           'tab:red',  'tab:purple', 'tab:olive',
                           'tab:cyan', 'tab:pink',   'tab:brown', 'tab:gray'])
        if n > colors.size:
            colors = np.random.choice(colors, size=(n,), replace=True, p=None)
        axis = self.ax_traj
        traj = self.traj_vals
        traj_des = self.traj_des_vals
        X = self.traj_t
        X_des = self.traj_des_t
        axis.clear()

        for i in range(n):
            axis.plot(X, traj[i, :].flatten(), color=colors[i],   label=str(i))

        for i in range(n):
            axis.plot(X_des, traj_des[i, :].flatten(), color=colors[i], linestyle='dashed',  label='des'+str(i))

        axis.set_title(r'$\theta$ trajectories')
        axis.legend()

    def plot_curr_est(self, i=None):
        n = self.num_links

        X = [e for e in range(0, n)]
        axis = self.ax_curr_est
        param_errors = self.est_vals
        axis[0, 0].clear()
        try:
            Y = param_errors[0:n, -1]
        except:
            return
        axis[0, 0].stem(X, Y)
        axis[0, 0].set_title(r'$\Delta$$\theta$')

        axis[0, 1].clear()
        Y = param_errors[n:2*n, -1]
        axis[0, 1].stem(X, Y)
        axis[0, 1].set_title(r'$\Delta$d')

        axis[1, 0].clear()
        Y = param_errors[2*n:3*n, -1]
        axis[1, 0].stem(X, Y)
        axis[1, 0].set_title(r'$\Delta$a')

        axis[1, 1].clear()
        Y = param_errors[3*n:4*n, -1]
        axis[1, 1].stem(X, Y)
        axis[1, 1].set_title(r'$\Delta$$\alpha$')

    def publish_est(self):
        self.plot_est()
        self.fig_est.canvas.draw()
        image_from_plot = np.frombuffer(self.fig_est.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(self.fig_est.canvas.get_width_height()[::-1] + (3,))
        self.pub_plot_est.publish(ros_numpy.msgify(Image, image_from_plot.astype(np.uint8), encoding='rgb8'))  # convert opencv image to ROS

    def publish_traj(self):
        self.plot_traj()
        self.fig_traj.canvas.draw()
        image_from_plot = np.frombuffer(self.fig_traj.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(self.fig_traj.canvas.get_width_height()[::-1] + (3,))
        self.pub_plot_traj.publish(ros_numpy.msgify(Image, image_from_plot.astype(np.uint8), encoding='rgb8')) # convert opencv image to ROS

    def save_data_shutdown(self):
        currtime = time.time()
        Path.cwd().joinpath('saved_plots/img').mkdir(parents=True, exist_ok=True)
        Path.cwd().joinpath('saved_plots/csv').mkdir(parents=True, exist_ok=True)

        savefile_name = str(Path.cwd().joinpath('saved_plots/img', 'estimate_{}.png'.format(int(currtime))))
        self.plot_est()
        self.fig_est.canvas.draw()
        self.fig_est.savefig(savefile_name)

        savefile_name = str(Path.cwd().joinpath('saved_plots/csv', 'estimate_{}.csv'.format(int(currtime))))
        pd.DataFrame(self.est_vals).to_csv(savefile_name)

        savefile_name = str(Path.cwd().joinpath('saved_plots/img', 'traj_{}.png'.format(int(currtime))))
        self.plot_traj()
        self.fig_traj.canvas.draw()
        self.fig_traj.savefig(savefile_name)

        savefile_name = str(Path.cwd().joinpath('saved_plots/csv', 'traj_{}.csv'.format(int(currtime))))
        pd.DataFrame(self.traj).to_csv(savefile_name)
        rospy.loginfo("Saved data")


# Main function.
if __name__ == "__main__":
    rospy.init_node('data_visualizer')   # init ROS node named aruco_detector
    rospy.loginfo('#Node data_visualizer running#')
    while not rospy.get_rostime():      # wait for ros time service
        pass
    rospy.loginfo("Ros Time service is available")
    rospy.loginfo("Waiting for message on topic /iiwa_q")
    _msg = rospy.wait_for_message("iiwa_q", Array_f64)  # listen for first message on topic
    rospy.loginfo("Received message on topic /iiwa_q")

    num_joints = np.array(_msg.data).size  # determine number of parameters

    dv = DataVisualizer(num_joints)          # create instance
    rospy.on_shutdown(dv.save_data_shutdown)

    rate = rospy.Rate(10)  # ROS Rate at ... Hz

    while not rospy.is_shutdown():
        rate.sleep()






