import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from parameter_estimator import ParameterEstimator
from mpl_toolkits.mplot3d import axes3d
import utils
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from Pose_Estimation_Class import UKF, IEKF
from helpers import Tools

theta_nom = ParameterEstimator.dhparams["theta_nom"]
r_nom = ParameterEstimator.dhparams["r_nom"]
d_nom = ParameterEstimator.dhparams["d_nom"]
alpha_nom = ParameterEstimator.dhparams["alpha_nom"]

observations_file_str = "/home/armin/catkin_ws/src/kident2/src/observations.p"
observations_file = open(observations_file_str, 'rb')

# dump information to that file
observations = pickle.load(observations_file)
# close the file
observations_file.close()

X = list()
Y = list()
Z = list()
U = list()
V = list()
W = list()

X_m = list()
Y_m = list()
Z_m = list()
U_m = list()
V_m = list()
W_m = list()


ukf = UKF()
T_corr = np.array([[ 0,  0, 1, 0],
                   [-1,  0, 0, 0],
                   [ 0, -1, 0, 0],
                   [ 0,  0, 0, 1]]) # euler [ x: -np.pi/2, y: np.pi/2, z: 0 ]

T_W0 = np.array([[-1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, 1, 0.36],
                 [0, 0, 0, 1]])

T_7C = utils.Trans(0, 0, 0.281) @ utils.Rz(np.pi)
tm = TransformManager()
marker_distance_errors = []
for id in observations:
# for id in [69]:
    for obs1 in observations[id]:
        obs2 = random.choice(observations[id])
        T_07_1 = ParameterEstimator.get_T_jk(0, 7, np.array(obs1['q']), theta_nom, d_nom, r_nom, alpha_nom)
        tm.add_transform(f"7_1", "0", T_07_1)
        T_07_2 = ParameterEstimator.get_T_jk(0, 7, np.array(obs2['q']), theta_nom, d_nom, r_nom, alpha_nom)
        tm.add_transform(f"7_2", "0", T_07_2)
        T_W7_1 = T_W0 @ T_07_1
        T_W7_2 = T_W0 @ T_07_2
        tm.add_transform("0", "W", T_W0)
        T_CM_1 = T_corr @ utils.H_rvec_tvec(obs1["rvec"], obs1["tvec"])
        T_CM_2 = T_corr @ utils.H_rvec_tvec(obs2["rvec"], obs2["tvec"])
        tm.add_transform("M_1", "C_1", T_CM_1)
        tm.add_transform("M_2", "C_2", T_CM_2)

        tm.add_transform("C_1", "7_1", T_7C)
        tm.add_transform("C_2", "7_2", T_7C)
        T_WC_1 = T_W7_1 @ T_7C
        T_WC_2 = T_W7_2 @ T_7C


        T_WM_1 = T_W7_1 @ T_7C @ T_CM_1
        T_WM_2 = T_W7_2 @ T_7C @ T_CM_2

        A = np.linalg.inv(T_W7_1) @ T_W7_2
        B = T_CM_1 @ np.linalg.inv(T_CM_2)
        current_error = np.linalg.norm(T_WM_1[0:3, 3] - T_WM_2[0:3, 3])
        marker_distance_errors.append(current_error)
        ########### PLOTS ################
        if False: #current_error > 0.025:
            fig = plt.figure()
            ax = tm.plot_frames_in("W", s=0.15)
            # ax = fig.add_subplot(111, projection='3d')
            ax.set_title("UKF add observation")

            X = [T_W7_1[0, 3], T_W7_2[0, 3]]
            Y = [T_W7_1[1, 3], T_W7_2[1, 3]]
            Z = [T_W7_1[2, 3], T_W7_2[2, 3]]
            ax.scatter(X, Y, Z, c='green')
            for x, y, z in zip(X, Y, Z):
                ax.plot([0, x], [0, y], [0, z], c="orange")


            X_c = [T_WC_1[0, 3], T_WC_2[0, 3]]
            Y_c = [T_WC_1[1, 3], T_WC_2[1, 3]]
            Z_c = [T_WC_1[2, 3], T_WC_2[2, 3]]
            ax.scatter(X_c, Y_c, Z_c, c='blue')
            for x, y, z, x_c, y_c, z_c in zip(X, Y, Z, X_c, Y_c, Z_c):
                ax.plot([x_c, x], [y_c, y], [z_c, z], c="yellow")


            X_m = [T_WM_1[0, 3], T_WM_2[0, 3]]
            Y_m = [T_WM_1[1, 3], T_WM_2[1, 3]]
            Z_m = [T_WM_1[2, 3], T_WM_2[2, 3]]
            ax.scatter(X_m, Y_m, Z_m, c='red')
            for x_c, y_c, z_c, x_m, y_m, z_m in zip(X_c, Y_c, Z_c, X_m, Y_m, Z_m):
                ax.plot([x_c, x_m], [y_c, y_m], [z_c, z_m], c="grey")
            # for x_m, y_m, z_m in zip(X_m, Y_m, Z_m):
            #     ax.plot([2, x_m], [2, y_m], [0, z_m], c="yellow")



            # #
            ##################################


            ax.plot([T_W7_1[0, 3], (T_W7_1 @ A)[0, 3]], [T_W7_1[1, 3], (T_W7_1 @ A)[1, 3]],
                    [T_W7_1[2, 3], (T_W7_1 @ A)[2, 3]], c="red")
            ax.plot([T_WC_1[0, 3], (T_WC_1 @ B)[0, 3]], [T_WC_1[1, 3], (T_WC_1 @ B)[1, 3]],
                    [T_WC_1[2, 3], (T_WC_1 @ B)[2, 3]], c="purple")

            utils.roundprint(A @ T_7C)
            utils.roundprint(T_7C @ B)
            plt.show()
        if current_error < 0.001:
            ukf.Update(B, A)
            # ukf.Update(np.linalg.inv(A), B)
            # ukf.Update(A, np.linalg.inv(B))
            # ukf.Update(np.linalg.inv(A), np.linalg.inv(B))


theta = np.linalg.norm(ukf.x[:3])
EPS = 0.00001
if theta < EPS:
    k = [0, 1, 0]  # VRML standard
else:
    k = ukf.x[0:3] / np.linalg.norm(ukf.x[:3])
euler_ukf = Tools.mat2euler(Tools.vec2rotmat(theta, k))
print('\n')
print('.....UKF Results')
print("UKF [euler_rpy(deg) , pos(mm)]:", np.array([euler_ukf]) * 180 / np.pi, ukf.x[3:] * 100)

plt.figure("UKF consistency")
plt.plot(range(len(ukf.consistency)), ukf.consistency)

plt.figure("Distance Error")
plt.plot(range(len(marker_distance_errors)), marker_distance_errors)
plt.show()