import pickle
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from parameter_estimator import ParameterEstimator
from mpl_toolkits.mplot3d import axes3d
import utils
from itertools import combinations
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

theta_nom = ParameterEstimator.dhparams["theta_nom"].astype(float)
r_nom = ParameterEstimator.dhparams["r_nom"].astype(float)
d_nom = ParameterEstimator.dhparams["d_nom"].astype(float)
alpha_nom = ParameterEstimator.dhparams["alpha_nom"].astype(float)

theta_error = np.array([0, 0, 0, 0, 0, - 0.0003, 0])
r_error = np.array([0, 0, 0, 0.004, 0, 0, 0])
d_error = np.array([0, 0, 0, 0.005, 0, 0, 0])
alpha_error = np.array([0, 0.0007, 0, 0, 0, 0, 0])

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

        # perform necessary inversions
        T_C7 = np.linalg.inv(T_7C)
        T_MC_2 = np.linalg.inv(T_CM_2)
        T_70_1 = np.linalg.inv(T_07_1)

        D_meas = T_7C @ T_CM_1 @ T_MC_2 @ T_C7
        D_nom = T_70_1 @ T_07_2
        delta_D = D_meas @ np.linalg.inv(D_nom) - np.eye(4)
        drvec, _ = cv2.Rodrigues(delta_D[0:3, 0:3] + np.eye(3))
        drvec = drvec.flatten()
        # drvec = np.array([delta_D[2, 1], delta_D[0, 2], delta_D[1, 0]])
        dtvec = delta_D[0:3, 3]
        pose_error = np.concatenate((dtvec, drvec))

        # calculate the corresponding difference jacobian
        jacobian = pe.get_parameter_jacobian_dual_2(q1=q1, q2=q2,
                                                  theta_all=pe.theta_nom,
                                                  d_all=pe.d_nom,
                                                  r_all=pe.r_nom,
                                                  alpha_all=pe.alpha_nom)

        # calculate position error in data
        diff = np.linalg.norm(np.array([(T_07_1 @ T_7C @ T_CM_1)[0, 3] - (T_07_2 @ T_7C @ T_CM_2)[0, 3],
                         (T_07_1 @ T_7C @ T_CM_1)[1, 3] - (T_07_2 @ T_7C @ T_CM_2)[1, 3],
                         (T_07_1 @ T_7C @ T_CM_1)[2, 3] - (T_07_2 @ T_7C @ T_CM_2)[2, 3]]))
        if diff > 0.02:
            continue
        diff_k.append(diff)

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
# plt.show()
for elem in estimate_k:
    print(elem)

plt.figure('diffs')
plt.plot(diff_k)
mean = np.mean(diff_k)
print(f"mean diff = {mean}")

fig_curr_est, ax_curr_est = plt.subplots(2, 2)
n = 7
X = [e for e in range(0, n)]
axis = ax_curr_est
param_errors = estimates_k
axis[0, 0].clear()
Y = param_errors[0:n, -1]
axis[0, 0].stem(X, Y)
axis[0, 0].set_title(r'$\Delta$$\theta$')

axis[0, 1].clear()
Y = param_errors[n:2*n, -1]
axis[0, 1].stem(X, Y)
axis[0, 1].set_title(r'$\Delta$d')

axis[1, 0].clear()
Y = param_errors[2*n:3*n, -1]
axis[1, 0].stem(X, Y)
axis[1, 0].set_title(r'$\Delta$r')

axis[1, 1].clear()
Y = param_errors[3*n:4*n, -1]
axis[1, 1].stem(X, Y)
axis[1, 1].set_title(r'$\Delta$$\alpha$')

plt.show()
print("done")