import pickle
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m
from parameter_estimator import ParameterEstimator
from mpl_toolkits.mplot3d import axes3d
import utils
from itertools import combinations
from scipy.spatial.transform import Rotation
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

theta_nom = ParameterEstimator.dhparams["theta_nom"].astype(float)
r_nom = ParameterEstimator.dhparams["r_nom"].astype(float)
d_nom = ParameterEstimator.dhparams["d_nom"].astype(float)
alpha_nom = ParameterEstimator.dhparams["alpha_nom"].astype(float)

theta_error = np.array([0, 0, 0, 0, 0, 0, 0])
r_error = np.array([0, 0, 0, 0, 0.02, 0, 0])
d_error = np.array([0, 0, 0, 0, 0, 0, 0])
alpha_error = np.array([0, 0, 0, 0, 0, 0, 0])

# r_error = np.hstack((np.zeros(1), np.random.normal(loc=0, scale=0.01, size=(6,))))
# d_error = np.hstack((np.zeros(1), np.random.normal(loc=0, scale=0.01, size=(6,))))
# alpha_error = np.hstack((np.zeros(1), np.random.normal(loc=0, scale=0.01, size=(6,))))
# theta_error = np.hstack((np.zeros(1), np.random.normal(loc=0, scale=0.01, size=(6,))))
# print(r_error)

r_nom = r_nom + r_error
theta_nom = theta_nom + theta_error
alpha_nom = alpha_nom + alpha_error
d_nom = d_nom + d_error


r_nom = r_nom + r_error
theta_nom = theta_nom + theta_error
alpha_nom = alpha_nom + alpha_error
d_nom = d_nom + d_error

observations_file_str = "observations_fake.p"
observations_file = open(observations_file_str, 'rb')
# dump information to that file
observations = pickle.load(observations_file)
# close the file
observations_file.close()

observations_file_str = "observations_fake_marker_pos.p"
observations_file = open(observations_file_str, 'rb')
# dump information to that file
T_WMs = pickle.load(observations_file)
# close the file
observations_file.close()

pe = ParameterEstimator()
estimates_k = np.empty((28, 0))

T_corr = np.array([[ 0,  0, 1, 0],
                   [-1,  0, 0, 0],
                   [ 0, -1, 0, 0],
                   [ 0,  0, 0, 1]]) # euler [ x: -np.pi/2, y: np.pi/2, z: 0 ]

T_W0 = np.array([[-1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, 1, 0.36],
                 [0, 0, 0, 1]])

T_7C = utils.Trans(0, 0, 0.281) @ utils.Rz(np.pi)

diff_k = list()
for markerid in list(observations)[:]:
    num_observed = 0
    print(f"working on marker {markerid}")
    for obs1, obs2 in combinations(observations[markerid], 2):
        if num_observed > 50:
            continue

        # extract measurements
        q1 = np.array(obs1["q"])
        q2 = np.array(obs2["q"])
        T_CM_1 = T_corr @ utils.H_rvec_tvec(obs1["rvec"], obs1["tvec"])
        T_CM_2 = T_corr @ utils.H_rvec_tvec(obs2["rvec"], obs2["tvec"])

        # calculate nominal transforms
        T_07_1 = pe.get_T_jk(0, 7, q1, theta_nom, d_nom, r_nom, alpha_nom)
        T_07_2 = pe.get_T_jk(0, 7, q2, theta_nom, d_nom, r_nom, alpha_nom)
        T_W7_1 = T_W0 @ T_07_1
        T_W7_2 = T_W0 @ T_07_2

        # perform necessary inversions
        T_C7 = np.linalg.inv(T_7C)
        T_MC_1 = np.linalg.inv(T_CM_1)
        T_MC_2 = np.linalg.inv(T_CM_2)
        T_70_1 = np.linalg.inv(T_07_1)
        T_70_2 = np.linalg.inv(T_07_2)

        # measurements
        # T_meas = T_7C @ T_CM_1 @ T_MC_2 @ T_C7
        # rmeas, tmeas = utils.mat2rvectvec(T_meas)

        # nominal
        # T_nom = T_70_1 @ T_07_2
        # rnom, tnom = utils.mat2rvectvec(T_nom)

        # difference
        # pose_error = np.concatenate((tmeas - tnom, rmeas - rnom))

        # pose difference
        t_M_A = (T_MC_1 @ T_C7)[0:3, 3]
        t_M_B = (T_MC_2 @ T_C7)[0:3, 3]
        t_W_A = T_W7_1[0:3, 3]
        t_W_B = T_W7_2[0:3, 3]
        _t_1 = t_M_B - t_M_A
        _t_2 = t_W_B - t_W_A
        t_diff = t_M_B - t_M_A - t_W_B + t_W_A

        # rotation difference
        R_M_A = (T_MC_1 @ T_C7)[0:3, 0:3]
        _rot = Rotation.from_matrix(R_M_A)
        r_M_A = _rot.as_euler("xyz", degrees=False)

        R_M_B = (T_MC_2 @ T_C7)[0:3, 0:3]
        _rot = Rotation.from_matrix(R_M_B)
        r_M_B = _rot.as_euler("xyz", degrees=False)

        R_0_A = T_07_1[0:3, 0:3]
        _rot = Rotation.from_matrix(R_0_A)
        r_0_A = _rot.as_euler("xyz", degrees=False)

        R_0_B = T_07_2[0:3, 0:3]
        _rot = Rotation.from_matrix(R_0_B)
        r_0_B = _rot.as_euler("xyz", degrees=False)

        R_0_A = T_07_1[0:3, 0:3]

        _r_1 = r_M_B - r_M_A
        _r_2 = r_0_B - r_0_A
        r_diff = r_M_B - r_M_A - r_0_B + r_0_A

        # pose error
        pose_error = np.concatenate((t_diff, r_diff))

        # jacobian
        jacobian_A = pe.get_parameter_jacobian_single(q=q1,
                                                        theta_all=pe.theta_nom,
                                                        d_all=pe.d_nom,
                                                        r_all=pe.r_nom,
                                                        alpha_all=pe.alpha_nom)
        jacobian_B = pe.get_parameter_jacobian_single(q=q2,
                                                      theta_all=pe.theta_nom,
                                                      d_all=pe.d_nom,
                                                      r_all=pe.r_nom,
                                                      alpha_all=pe.alpha_nom)
        _jacobian_1 = np.concatenate((R_0_B @ jacobian_B[0:3, :], R_0_B @ jacobian_B[3:6, :]), axis=0)
        _jacobian_2 = np.concatenate((R_0_A @ jacobian_A[0:3, :], R_0_A @ jacobian_A[3:6, :]), axis=0)
        jacobian = _jacobian_1 - _jacobian_2

        # calculate position error in data
        # diff = np.linalg.norm(np.array([(T_07_1 @ T_7C @ T_CM_1)[0, 3] - (T_07_2 @ T_7C @ T_CM_2)[0, 3],
        #                  (T_07_1 @ T_7C @ T_CM_1)[1, 3] - (T_07_2 @ T_7C @ T_CM_2)[1, 3],
        #                  (T_07_1 @ T_7C @ T_CM_1)[2, 3] - (T_07_2 @ T_7C @ T_CM_2)[2, 3]]))
        # if diff > 0.02:
        #     continue
        # diff_k.append(diff)

        pe.rls.add_obs(S=jacobian, Y=pose_error)
        estimate_k = pe.rls.get_estimate().flatten()
        estimate_k = np.reshape(np.array(estimate_k), (-1, 1))
        estimates_k = np.hstack((estimates_k, estimate_k))
        num_observed += 1

fig_est, ax_est = plt.subplots(2, 2)
fig_est.set_size_inches(16, 9, forward=True)
fig_est.tight_layout(pad=2)
n = 7
n_tot = 4*7

colors = np.array(['tab:blue', 'tab:orange', 'tab:green',
                   'tab:red',  'tab:purple', 'tab:olive',
                   'tab:cyan', 'tab:pink',   'tab:brown', 'tab:gray'])
if n > colors.size:
    colors = np.random.choice(colors, size=(n,), replace=True, p=None)

axis = ax_est
X = list(range(len(estimates_k[0, :])))
param_errors = estimates_k
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

for elem in estimate_k:
    print(elem)

plt.figure('diffs')
plt.plot(diff_k)
# mean = np.mean(diff_k)
# print(f"mean diff = {mean}")

fig_curr_est, ax_curr_est = plt.subplots(2, 2)
n = 7
X = [e for e in range(0, n)]
axis = ax_curr_est
param_errors = estimates_k
axis[0, 0].clear()
Y = param_errors[0:n, -1]
axis[0, 0].scatter(X, -theta_error, c='orange', s=100)
axis[0, 0].stem(X, Y)
axis[0, 0].set_title(r'$\Delta$$\theta$')

axis[0, 1].clear()
Y = param_errors[n:2*n, -1]
axis[0, 1].scatter(X, -d_error, c='orange', s=100)
axis[0, 1].stem(X, Y)
axis[0, 1].set_title(r'$\Delta$d')

axis[1, 0].clear()
Y = param_errors[2*n:3*n, -1]
axis[1, 0].scatter(X, -r_error, c='orange', s=100)
axis[1, 0].stem(X, Y)
axis[1, 0].set_title(r'$\Delta$r')

axis[1, 1].clear()
Y = param_errors[3*n:4*n, -1]
axis[1, 1].scatter(X, -alpha_error, c='orange', s=100)
axis[1, 1].stem(X, Y)
axis[1, 1].set_title(r'$\Delta$$\alpha$')

plt.show()
print("done")