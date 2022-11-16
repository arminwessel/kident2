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

theta_nom = ParameterEstimator.dhparams["theta_nom"]
print(theta_nom)
r_nom = ParameterEstimator.dhparams["r_nom"]
d_nom = ParameterEstimator.dhparams["d_nom"]
alpha_nom = ParameterEstimator.dhparams["alpha_nom"]

observations_file_str = "observations_big.p"
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

# T_7C = np.array([[-1, 0, 0, 0],
#                  [0, -1, 0, 0],
#                  [0, 0, 1, 0.435],
#                  [0, 0, 0, 1]])
T_7C = utils.Trans(0, 0, 0.281) @ utils.Rz(np.pi)

# T_corr = np.array([[ 0,  0, 1, 0],
#                    [-1,  0, 0, 0],
#                    [ 0, -1, 0, 0],
#                    [ 0,  0, 0, 1]]) # euler [ x: -np.pi/2, y: np.pi/2, z: 0 ]

T_corr = np.array([[ 0,  0, 1, 0],
                   [ -1,  0, 0, 0],
                   [ 0, -1, 0, 0],
                   [ 0,  0, 0, 1]])

# T_0C = np.array([  [1, 0, 0, 1],
#                    [0, 1, 0, 1],
#                    [0, 0, 1, 1],
#                    [0, 0, 0, 1]])
# T_0C = pt.transform_from(pr.matrix_from_compact_axis_angle([0, -0.3, 0]), [1, 1, 1])
# for id in observations.keys():
# for id in random.sample(observations.keys(), 1):
#     print(id)

tm = TransformManager()

for markerid in list(observations):
    for id, obs in enumerate(observations[markerid]):
        # T_07 = ParameterEstimator.get_T__i0(7, np.array(obs['q']), d_nom, r_nom, alpha_nom)

        # obs['q'] = [0, 0, 0, 0, 0, 0, 0]

        # tm.add_transform(f"{id}_frame0", "W", T_W0)
        #
        # T_01 = ParameterEstimator.get_T_jk(0, 1, np.array(obs['q']), theta_nom, d_nom, r_nom, alpha_nom)
        # tm.add_transform(f"{id}_frame1", f"{id}_frame0", T_01)
        #
        # T_12 = ParameterEstimator.get_T_jk(1, 2, np.array(obs['q']), theta_nom, d_nom, r_nom, alpha_nom)
        # tm.add_transform(f"{id}_frame2", f"{id}_frame1", T_12)
        #
        # T_23 = ParameterEstimator.get_T_jk(2, 3, np.array(obs['q']), theta_nom, d_nom, r_nom, alpha_nom)
        # tm.add_transform(f"{id}_frame3", f"{id}_frame2", T_23)
        #
        # T_34 = ParameterEstimator.get_T_jk(3, 4, np.array(obs['q']), theta_nom, d_nom, r_nom, alpha_nom)
        # tm.add_transform(f"{id}_frame4", f"{id}_frame3", T_34)
        #
        # T_45 = ParameterEstimator.get_T_jk(4, 5, np.array(obs['q']), theta_nom, d_nom, r_nom, alpha_nom)
        # tm.add_transform(f"{id}_frame5", f"{id}_frame4", T_45)
        #
        # T_56 = ParameterEstimator.get_T_jk(5, 6, np.array(obs['q']), theta_nom, d_nom, r_nom, alpha_nom)
        # tm.add_transform(f"{id}_frame6", f"{id}_frame5", T_56)
        #
        # T_67 = ParameterEstimator.get_T_jk(6, 7, np.array(obs['q']), theta_nom, d_nom, r_nom, alpha_nom)
        # tm.add_transform(f"{id}_frame7", f"{id}_frame6", T_67)
        #
        # tm.add_transform(f"{id}_frameC", f"{id}_frame7", T_7C)

        # T_WC = T_W0 @ T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56 @ T_67 @ T_7C
        T_07 = ParameterEstimator.get_T_jk(0, 7, obs['q'], theta_nom, d_nom, r_nom, alpha_nom)
        T_WC = T_W0 @ T_07 @ T_7C
        trans_z = T_WC @ np.array([0, 0, 1, 1])
        X.append(T_WC[0, 3])
        Y.append(T_WC[1, 3])
        Z.append(T_WC[2, 3])
        U.append(trans_z[0])
        V.append(trans_z[1])
        W.append(trans_z[2])

        T_CM = utils.H_rvec_tvec(obs["rvec"], obs["tvec"])
        T_CM_corr = T_corr @ T_CM
        # tm.add_transform(f"{id}_frameM", f"{id}_frameC", T_CM_corr)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Observations")


ax.scatter(X, Y, Z, c='green')
# ax.scatter([1], [1], [1], c='green')
ax.scatter(X_m, Y_m, Z_m, c='red')
# ax.scatter([0.7670], [1.5403], [2.4575], c='purple')
# ax.scatter([0], [0], [0], c='blue')
#
#
# # ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
# # ax.quiver(X_m, Y_m, Z_m, U_m, V_m, W_m, length=0.1, normalize=True)
#
for x, y, z, x_m, y_m, z_m in zip(X, Y, Z, X_m, Y_m, Z_m):
    ax.plot([x, x_m], [y, y_m], [z, z_m], c="grey")

# for x_m, y_m, z_m in zip(X_m, Y_m, Z_m):
#     ax.plot([2, x_m], [2, y_m], [0, z_m], c="yellow")

for x, y, z in zip(X, Y, Z):
    ax.plot([0, x], [0, y], [0, z], c="orange")
# #
# # ax.scatter([2.4575, 2.5353, 2.2915], [0, 0.9822, 0.8877], [1.7207, 1.2679, 1.7207], c="black")
# # ax.scatter([2.7189], [0], [1.2679], c="purple")
# # Create a sphere
# r = 0.82
# pi = np.pi
# cos = np.cos
# sin = np.sin
# phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
# sphere_x = r*sin(phi)*cos(theta)
# sphere_y = r*sin(phi)*sin(theta)
# sphere_z = r*cos(phi)
#
#
# # ax.plot_surface(
# #     sphere_x, sphere_y, sphere_z,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0.3, edgecolor = 'k')
#
plt.show()



