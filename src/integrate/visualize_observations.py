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
import pandas as pd
from scipy.spatial.transform import Rotation


l_to_df = []
theta_nom = ParameterEstimator.dhparams["theta_nom"]
print(theta_nom)
r_nom = ParameterEstimator.dhparams["r_nom"]
d_nom = ParameterEstimator.dhparams["d_nom"]
alpha_nom = ParameterEstimator.dhparams["alpha_nom"]

observations_file_str = 'obs_2007_gazebo_iiwa_stopping.bag_20230720-135812.p'
#observations_file_str = "obs_single_20230724-124651.p"

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

X_me = list()
Y_me = list()
Z_me = list()
U_me = list()
V_me = list()
W_me = list()

T_W0 = np.array([[-1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, 1, 0.36],
                 [0, 0, 0, 1]])

T_8C = np.eye(4)


T_corr = np.array([[ 0,  0, 1, 0],
                   [ -1, 0, 0, 0],
                   [ 0, -1, 0, 0],
                   [ 0,  0, 0, 1]])


tm = TransformManager()
list_observations = list(observations)
for markerid in list_observations:
    for obs in observations[markerid]:
        q = np.hstack((np.array(obs["q"]), np.zeros(1)))
        T_08 = ParameterEstimator.get_T_jk(0, 8, q, theta_nom, d_nom, r_nom, alpha_nom)
        T_W8 = T_W0 @ T_08
        T_WC = T_W0 @ T_08 @ T_8C
        trans_z = T_WC @ np.array([0, 0, 1, 1])
        X.append(T_WC[0, 3])
        Y.append(T_WC[1, 3])
        Z.append(T_WC[2, 3])
        U.append(trans_z[0])
        V.append(trans_z[1])
        W.append(trans_z[2])

        T_CM = utils.H_rvec_tvec(obs["rvec"], obs["tvec"])
        T_CM_corr = T_corr @ T_CM @ np.linalg.inv(T_corr) @ utils.Ry(-np.pi/2)

        M = T_WC @ T_CM_corr

        trans_z_m = M @ np.array([0, 0, 1, 1])
        X_m.append(M[0, 3])
        Y_m.append(M[1, 3])
        Z_m.append(M[2, 3])
        U_m.append(trans_z_m[0])
        V_m.append(trans_z_m[1])
        W_m.append(trans_z_m[2])

        # trans_z_me = M @ np.array([0, 0, 1, 1])
        # X_me.append(M_err[0, 3])
        # Y_me.append(M_err[1, 3])
        # Z_me.append(M_err[2, 3])
        # U_me.append(trans_z_me[0])
        # V_me.append(trans_z_me[1])
        # W_me.append(trans_z_me[2])
        # ax = tm.plot_frames_in("W", s=0.15) # whitelist=["W", "0_frame0", "0_frame3", "0_frame5", "0_frameC", "0_frameM"]
        # ax.set_xlim((-0.25, 3))
        # ax.set_ylim((-0.25, 3))
        # ax.set_zlim((-0.25, 3))
        # ax.scatter([0.7670], [1.5403], [2.4575], c='purple')
        # if id==3:
        #     plt.show()
        #     break

        _r = Rotation.from_matrix(M[0:3, 0:3])
        Mrx, Mry, Mrz = _r.as_euler('XYZ')

        _r = Rotation.from_matrix(T_W8[0:3, 0:3])
        Erx, Ery, Erz = _r.as_euler('XYZ')

        _r = Rotation.from_matrix(T_WC[0:3, 0:3])
        Crx, Cry, Crz = _r.as_euler('XYZ')

        _r = Rotation.from_matrix(T_CM[0:3, 0:3])
        Cmrx, Cmry, Cmrz = _r.as_euler('XYZ')
        l_to_df.append({
            'markerid': markerid,
            't': obs['t'],
            'q': q,
            'T_W8': T_W8,
            'T_WC': T_WC,
            'M': M,
            'Mx': M[0, 3],
            'My': M[1, 3],
            'Mz': M[2, 3],
            'Mrx': Mrx,
            'Mry': Mry,
            'Mrz': Mrz,
            'Ex': T_W8[0, 3],
            'Ey': T_W8[1, 3],
            'Ez': T_W8[2, 3],
            'Erx': Erx,
            'Ery': Ery,
            'Erz': Erz,
            'Cx': T_WC[0, 3],
            'Cy': T_WC[1, 3],
            'Cz': T_WC[2, 3],
            'Crx': Crx,
            'Cry': Cry,
            'Crz': Crz,
            'Cmx': T_CM[0, 3],
            'Cmy': T_CM[1, 3],
            'Cmz': T_CM[2, 3],
            'Cmrx': Cmrx,
            'Cmry': Cmry,
            'Cmrz': Cmrz
        })

df = pd.DataFrame(l_to_df)
df.to_csv(observations_file_str + '.csv')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Observations")
#
#
# ax.scatter(X, Y, Z, c='green')
#
# ax.scatter(X_m, Y_m, Z_m, c='red')
#
# for x, y, z, x_m, y_m, z_m in zip(X, Y, Z, X_m, Y_m, Z_m):
#     ax.plot([x, x_m], [y, y_m], [z, z_m], c="grey")
#
# for x, y, z in zip(X, Y, Z):
#     ax.plot([0, x], [0, y], [0, z], c="orange")
#
# plt.show()



