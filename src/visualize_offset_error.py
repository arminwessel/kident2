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
r_nom = ParameterEstimator.dhparams["r_nom"]
d_nom = ParameterEstimator.dhparams["d_nom"]
alpha_nom = ParameterEstimator.dhparams["alpha_nom"]

observations_file_str = "/src/observations_small.p"
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

X_c = list()
Y_c = list()
Z_c = list()
U_c = list()
V_c = list()
W_c = list()

X_err = list()
Y_err = list()
Z_err = list()
U_err = list()
V_err = list()
W_err = list()

T_7C = np.array([[0.7660444,  0.0000000,  0.6427876, 0.1],
                 [0.3213938,  0.8660254, -0.3830222, 0.13],
                 [-0.556670,  0.5000000,  0.6634139, 0.185],
                 [0,          0,          0,         1]])
T_7C_wrong = np.copy(T_7C)
T_7C_wrong[2, 3] = 0
# for id in observations.keys():
for id in random.sample(observations.keys(), 5):
    for obs in observations[id]:
        T_07 = ParameterEstimator.get_T_jk(0,7, np.array(obs['q']), theta_nom, d_nom, r_nom, alpha_nom)

        trans_z = T_07 @ np.array([0, 0, 1, 1])
        X.append(T_07[0, 3])
        Y.append(T_07[1, 3])
        Z.append(T_07[2, 3])
        U.append(trans_z[0])
        V.append(trans_z[1])
        W.append(trans_z[2])

        T_0C = T_07 @ T_7C
        T_0M = np.array([[-1, 0, 0, 2], [0, -1, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]])
        T_CM = np.linalg.inv(T_0C) @ T_0M


        # T_7C_wrong[0:3, 0:3] = np.array([[0.7071068, -0.0000000, 0.7071068],
        #                                  [0.3535534, 0.8660254, -0.3535534],
        #                                  [-0.6123725, 0.5000000, 0.6123725]])
        T_err = T_07 @ T_7C_wrong @ T_CM # simulate error in offset here

        # M = U@H
        trans_z_c = T_0C @ np.array([0, 0, 1, 1])
        X_c.append(T_0C[0, 3])
        Y_c.append(T_0C[1, 3])
        Z_c.append(T_0C[2, 3])
        U_c.append(trans_z_c[0])
        V_c.append(trans_z_c[1])
        W_c.append(trans_z_c[2])

        trans_z_err = T_err @ np.array([0, 0, 1, 1])
        X_err.append(T_err[0, 3])
        Y_err.append(T_err[1, 3])
        Z_err.append(T_err[2, 3])
        U_err.append(trans_z_err[0])
        V_err.append(trans_z_err[1])
        W_err.append(trans_z_err[2])

        # trans_z_m = M @ np.array([0, 0, 1, 1])
        # X_m.append(M[0, 3])
        # Y_m.append(M[1, 3])
        # Z_m.append(M[2, 3])
        # U_m.append(trans_z_m[0])
        # V_m.append(trans_z_m[1])
        # W_m.append(trans_z_m[2])




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Observations")
# maxset = set(np.concatenate((X, Y, Z, X_m, Y_m, Z_m)))
# maxelem = max(maxset)
# ticks = (maxelem*np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])).tolist()
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_zticks(ticks)
#
# ax.axes.set_xlim3d(left=-0.98*maxelem, right=0.98*maxelem)
# ax.axes.set_ylim3d(bottom=-0.98*maxelem, top=0.98*maxelem)
# ax.axes.set_zlim3d(bottom=-0.98*maxelem, top=0.98*maxelem)


# ax.scatter([x-x_m for (x, x_m) in zip(X, X_m)], [y-y_m for (y, y_m) in zip(Y, Y_m)], [z-z_m for (z, z_m) in zip(Z, Z_m)], c='blue')
ax.scatter(X, Y, Z, c='green')
ax.scatter(X_c, Y_c, Z_c, c='red')
ax.scatter(X_err, Y_err, Z_err, c='purple')
# ax.scatter([0], [0], [0], c='blue')
ax.scatter(T_0M[0, 3], T_0M[1, 3], T_0M[2, 3], c='magenta')
# ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
# ax.quiver(X_m, Y_m, Z_m, U_m, V_m, W_m, length=0.1, normalize=True)
ax.quiver(X_c, Y_c, Z_c, U_c, V_c, W_c, length=0.1, normalize=True)


# for x, y, z, x_m, y_m, z_m in zip(X, Y, Z, X_m, Y_m, Z_m):
#     ax.plot([x, x_m], [y, y_m], [z, z_m], c="grey")

for x, y, z, x_c, y_c, z_c in zip(X, Y, Z, X_c, Y_c, Z_c):
    ax.plot([x, x_c], [y, y_c], [z, z_c], c="grey")

for x_c, y_c, z_c in zip(X_c, Y_c, Z_c):
    ax.plot([2, x_c], [2, y_c], [0, z_c], c="yellow")

for x, y, z in zip(X, Y, Z):
    ax.plot([x, 0], [y, 0], [z, 0], c="orange")

# Create a sphere
r = 0.82
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
sphere_x = r*sin(phi)*cos(theta)
sphere_y = r*sin(phi)*sin(theta)
sphere_z = r*cos(phi)


# ax.plot_surface(
#     sphere_x, sphere_y, sphere_z,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0.3, edgecolor = 'k')

plt.show()




