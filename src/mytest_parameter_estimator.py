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
import pandas as pd
import random
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
f_acin_colors = open('acin_colors.p', 'rb')
acin_colors = pickle.load(f_acin_colors)
f_acin_colors.close()

theta_nom = ParameterEstimator.dhparams["theta_nom"].astype(float)
r_nom = ParameterEstimator.dhparams["r_nom"].astype(float)
d_nom = ParameterEstimator.dhparams["d_nom"].astype(float)
alpha_nom = ParameterEstimator.dhparams["alpha_nom"].astype(float)

# theta_error = np.array([0, 0, 0, 0, 0, 0, 0, 0])*np.pi/180
# r_error = np.array([0, 0, 0, 0, 0, 0, 0, 0])/1000
# d_error = np.array([0, 0, 0.05, 0, 0, 0, 0, 0])/1000
# alpha_error = np.array([0, 0, 0, 0, 0, 0, 0, 0])*np.pi/180

# theta_error = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0])*np.pi/180
# r_error = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0])/1000
# d_error = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])/1000
# alpha_error = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])*np.pi/180


# r_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)/500, np.zeros(1)))  # random error in range [-1mm, +1mm)
# d_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)/500, np.zeros(1)))  # random error in range [-1mm, +1mm)
# alpha_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)*np.pi/180*2, np.zeros(1)))  # random error in range [-1deg, +1deg)
# theta_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)*np.pi/180*2, np.zeros(1)))  # random error in range [-1deg, +1deg)

# r_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)/1000, np.zeros(1)))  # random error in range [-0.5mm, +0.5mm)
# d_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)/1000, np.zeros(1)))  # random error in range [-0.5mm, +0.5mm)
# alpha_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)*np.pi/180, np.zeros(1)))  # random error in range [-0.5deg, +0.5deg)
# theta_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)*np.pi/180, np.zeros(1)))  # random error in range [-0.5deg, +0.5deg)

# r_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)/5000, np.zeros(1)))  # random error in range [-0.1mm, +0.1mm)
# d_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)/5000, np.zeros(1)))  # random error in range [-0.1mm, +0.1mm)
# alpha_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)*np.pi/1800*2, np.zeros(1)))  # random error in range [-0.1deg, +0.1deg)
# theta_error = np.hstack((np.zeros(1), (np.random.rand(6)-np.ones(6)*0.5)*np.pi/1800*2, np.zeros(1)))  # random error in range [-0.1deg, +0.1deg)

# r_error = np.hstack((np.zeros(1), np.random.normal(loc=0, scale=sigma_r/1000, size=6), np.zeros(1)))  # random error in range [-0.1mm, +0.1mm)
# d_error = np.hstack((np.zeros(1), np.random.normal(loc=0, scale=sigma_d/1000, size=6), np.zeros(1)))  # random error in range [-0.1mm, +0.1mm)
# alpha_error = np.hstack((np.zeros(1), np.random.normal(loc=0, scale=sigma_alpha/180*np.pi, size=6), np.zeros(1)))  # random error in range [-0.1deg, +0.1deg)
# theta_error = np.hstack((np.zeros(1), np.random.normal(loc=0, scale=sigma_theta/180*np.pi, size=6), np.zeros(1)))  # random error in range [-0.1deg, +0.1deg)

sigma_r = 10  # in mm
sigma_d = 10  # in mm
sigma_alpha = 10  # in degrees
sigma_theta = 10  # in degrees

normal_dist = np.array([ 1.3046512, 0.96882621, 0.15600486, -0.51135905, -0.75806078, 0.82808355])

r_error = np.hstack((np.zeros(1), normal_dist*sigma_r/1000, np.zeros(1)))  # random error in range [-0.1mm, +0.1mm)
d_error = np.hstack((np.zeros(1), normal_dist*sigma_d/1000, np.zeros(1)))  # random error in range [-0.1mm, +0.1mm)
alpha_error = np.hstack((np.zeros(1), normal_dist*sigma_alpha/180*np.pi, np.zeros(1)))  # random error in range [-0.1deg, +0.1deg)
theta_error = np.hstack((np.zeros(1), normal_dist*sigma_theta/180*np.pi, np.zeros(1)))  # random error in range [-0.1deg, +0.1deg)

r_nom = r_nom + r_error
theta_nom = theta_nom + theta_error
alpha_nom = alpha_nom + alpha_error
d_nom = d_nom + d_error

observations_file_str = 'observations_fake2.p'
# observations_file_str = 'obs_2007_gazebo_iiwa_stopping.bag_20230720-135812.p'

observations_file = open(observations_file_str, 'rb')
# dump information to that file
observations = pickle.load(observations_file)
# close the file
observations_file.close()
#
# observations_file_str = "observations_fake_marker_pos.p"
# observations_file = open(observations_file_str, 'rb')
# # dump information to that file
# T_WMs = pickle.load(observations_file)
# # close the file
# observations_file.close()

# observations_file_str = "observations_small.p"
# observations_file = open(observations_file_str, 'rb')
# # dump information to that file
# observations = pickle.load(observations_file)
# # close the file
# observations_file.close()

pe = ParameterEstimator()


T_corr = np.array([[ 0,  0, 1, 0],
                   [-1,  0, 0, 0],
                   [ 0, -1, 0, 0],
                   [ 0,  0, 0, 1]]) # euler [ x: -np.pi/2, y: np.pi/2, z: 0 ]

T_W0 = np.array([[-1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, 1, 0.36],
                 [0, 0, 0, 1]])

# T_7C = utils.Trans(0, 0, 0.281) @ utils.Rz(np.pi)

diff_k = list()
current_marker = [None]
estimates_k = np.zeros((8 * 4 - 6, 1))


def residuals(params, errors_tot, jacobian_tot):
    ret = errors_tot - jacobian_tot @ params
    return ret
def identify(observations, theta, d, r, alpha , k_obs):
    global pe
    print("Starting loop over observations")
    start_time = time.time()

    # reset rls matrices and LM accumulators

    pe.rls.reset()
    jacobian_tot = np.zeros((0, 8 * 4 - 6))
    errors_tot = np.zeros(0)

    for markerid in list(observations)[:]:
        num_observed = 0
        # print(f"working on marker {markerid}")
        count = 0
        comparisons = []
        for obs1, obs2 in combinations(observations[markerid], 2):
            count = count+1
            comparisons.append((obs1, obs2))
        # print(f"total number of possible observations: {count}")
        for obs1, obs2 in random.choices(comparisons, k=k_obs):

            # extract measurements
            q1 = np.hstack((np.array(obs1["q"]), np.zeros(1)))
            q2 = np.hstack((np.array(obs2["q"]), np.zeros(1)))
            T_CM_1 = T_corr @ utils.H_rvec_tvec(obs1["rvec"], obs1["tvec"]) @ np.linalg.inv(T_corr) @ utils.Ry(-np.pi/2)
            T_CM_2 = T_corr @ utils.H_rvec_tvec(obs2["rvec"], obs2["tvec"]) @ np.linalg.inv(T_corr) @ utils.Ry(-np.pi/2)

            # calculate nominal transforms
            T_08_1 = pe.get_T_jk(0, 8, q1, theta, d, r, alpha)
            T_08_2 = pe.get_T_jk(0, 8, q2, theta, d, r, alpha)

            # perform necessary inversions
            # T_C7 = np.linalg.inv(T_7C)
            T_MC_2 = np.linalg.inv(T_CM_2)
            T_80_1 = np.linalg.inv(T_08_1)

            D_meas = T_CM_1 @ T_MC_2
            D_nom = T_80_1 @ T_08_2
            delta_D = D_meas @ np.linalg.inv(D_nom)
            delta_D_skew = 0.5 * (delta_D - delta_D.T)

            drvec = np.array([delta_D_skew[2, 1], delta_D_skew[0, 2], delta_D_skew[1, 0]])
            dtvec = delta_D[0:3, 3]
            pose_error = np.concatenate((dtvec, drvec))

            # calculate the corresponding difference jacobian
            jacobian = pe.get_parameter_jacobian_improved(q1=q1, q2=q2,
                                                      theta_all=theta,
                                                      d_all=d,
                                                      r_all=r,
                                                      alpha_all=alpha)

            # delete non-observable link zeros and the merged camera params
            jacobian = np.delete(jacobian, [0, 8, 16, 24, 7, 23], 1)

            # do rls computation
            # pe.rls.add_obs(S=jacobian, Y=pose_error)
            # estimate_k = pe.rls.get_estimate().flatten()
            # estimate_k = np.reshape(np.array(estimate_k), (-1, 1))
            # estimates_k = np.hstack((estimates_k, estimate_k))
            # num_observed += 1
            current_marker.append(markerid)

            # save jacobian to big jacobian
            # jacobian_tot = np.concatenate((jacobian_tot, jacobian, jacobian[3:6, :], jacobian[3:6, :], jacobian[3:6, :], jacobian[3:6, :], jacobian[3:6, :], jacobian[3:6, :]), axis=0)
            # errors_tot = np.concatenate((errors_tot, pose_error, pose_error[3:6], pose_error[3:6], pose_error[3:6], pose_error[3:6], pose_error[3:6], pose_error[3:6]), axis=0)
            jacobian_tot = np.concatenate((jacobian_tot, jacobian), axis=0)
            errors_tot = np.concatenate((errors_tot, pose_error), axis=0)
    print(f"Finished loop, it took {time.time() - start_time} seconds")
    # _temp = np.concatenate((jacobian_tot, np.reshape(errors_tot, (len(errors_tot), 1))), axis=1)
    # np.random.shuffle(_temp)
    # jacobian_tot_shuffled = _temp[:, :-1]  # 33rd colum is the error vector concat previously
    # errors_tot_shuffled = _temp[:, -1]


    print("Start LM Algorithm")
    start_time = time.time()
    res = least_squares(fun=residuals, x0=np.zeros(4*8-6), method='lm', args=(errors_tot, jacobian_tot))
    x_lm = res.x
    print(f"Finished LM, it took {time.time() - start_time} seconds")
    # return {'rls': estimates_k, 'lm': x_lm}
    return x_lm

theta = theta_nom
d = d_nom
r = r_nom
alpha = alpha_nom
x = identify(observations, theta, d, r, alpha, 50)
x = np.reshape(np.array(x), (-1, 1))
estimates_k = np.hstack((estimates_k, x))
print(f"theta: {theta}\nd: {d}\nr: {r}\nalpha: {alpha}\n\n")

for i in range(10):
    print(f"Iteration {i}")
    theta = theta + np.concatenate((np.zeros(1), x.flatten()[0:6], np.zeros(1)))
    d = d + np.concatenate((np.zeros(1), x.flatten()[6:13]))
    r = r + np.concatenate((np.zeros(1), x.flatten()[13:19], np.zeros(1)))
    alpha = alpha + np.concatenate((np.zeros(1), x.flatten()[19:26]))
    x = identify(observations, theta, d, r, alpha, 50)
    x = np.reshape(np.array(x), (-1, 1))
    estimates_k = np.hstack((estimates_k, x))
    print(f"theta: {theta}\nd: {d}\nr: {r}\nalpha: {alpha}\n\n")



fig_est, ax_est = plt.subplots(2, 2)
fig_est.set_size_inches(16, 9, forward=True)
fig_est.tight_layout(pad=2)
n = 8
n_tot = 4*8

colors = np.array(['tab:blue', 'tab:orange', 'tab:green',
                   'tab:red',  'tab:purple', 'tab:olive',
                   'tab:cyan', 'tab:pink',   'tab:brown', 'tab:gray'])
colors = np.array([acin_colors['acin_red'], acin_colors['acin_green'], acin_colors['acin_yellow'],
                   acin_colors['TU_blue'], acin_colors['acin_yellow_variant'],
                   acin_colors['acin_green_variant'], acin_colors['acin_blue_variant'],
                   acin_colors['TU_pink_variant']])
if n > colors.size:
    colors = np.random.choice(colors, size=(n,), replace=True, p=None)

axis = ax_est
X = list(range(len(estimates_k[0, :])))
param_errors = estimates_k
axis[0, 0].clear()
for i in range(6):
    Y = param_errors[i, :].flatten() * 180 / np.pi
    axis[0, 0].plot(X, Y, color=colors[i],   label=str(i))
axis[0, 0].set_title(r'$\Delta$$\theta [°]$')
axis[0, 0].legend()

axis[0, 1].clear()
for i in range(7):
    Y = param_errors[i+6, :].flatten() * 1000
    axis[0, 1].plot(X, Y, color=colors[i],   label=str(i))
axis[0, 1].set_title(r'$\Delta$d [mm]')
axis[0, 1].legend()

axis[1, 0].clear()
for i in range(6):
    Y = param_errors[i+13, :].flatten() * 1000
    axis[1, 0].plot(X, Y, color=colors[i],   label=str(i))
axis[1, 0].set_title(r'$\Delta$r [mm]')
axis[1, 0].legend()

axis[1, 1].clear()
for i in range(7):
    Y = param_errors[i+19, :].flatten() * 180 / np.pi
    axis[1, 1].plot(X, Y, color=colors[i],   label=str(i))
axis[1, 1].set_title(r'$\Delta$$\alpha [°]$')
axis[1, 1].legend()

fig_marker, ax_marker = plt.subplots(1,1)
X = list(range(len(current_marker)))
ax_marker.plot(X, current_marker, c=acin_colors['TU_blue'])

# for elem in estimate_k:
#     print(elem)
#
# plt.figure('diffs')
# plt.plot(diff_k)
# # mean = np.mean(diff_k)
# # print(f"mean diff = {mean}")

fig_curr_est, ax_curr_est = plt.subplots(2, 2, figsize=(6, 6))
fig_curr_est.tight_layout(pad=2)
n = 8
X = [e for e in range(0, n)]
axis = ax_curr_est

# param_errors = np.concatenate((theta_nom[1:7], d_nom[1:8], r_nom[1:7], alpha_nom[1:8])).reshape(-1,1)
param_errors = np.concatenate((theta[1:7], d[1:8], r[1:7], alpha[1:8])).reshape(-1,1) - np.concatenate((theta_nom[1:7], d_nom[1:8], r_nom[1:7], alpha_nom[1:8])).reshape(-1,1)
# param_errors = estimates_k[:, -1] - np.concatenate((theta_nom[1:7], np.zeros(13), alpha_nom[1:8])).reshape(-1,1)

axis[0, 0].clear()

Y1 = param_errors[0:6, -1] * 180 / np.pi
Z1 = -theta_error * 180 / np.pi
X1 = np.arange(1, len(Y1)+1)
_min = np.concatenate((Y1, Z1)).min()
_max = np.concatenate((Y1, Z1)).max()
_abs = max(abs(_min), abs(_max))
_min = _min + np.sign(_min)*0.1*_abs  # add a buffer of 10% of the max value on top and on bottom
_max = _max + np.sign(_max)*0.1*_abs
axis[0, 0].set_ylim([_min, _max])
axis[0, 0].scatter(X, Z1, c=acin_colors['acin_yellow'], s=100)
axis[0, 0].set_title(r'$\Delta$$\theta$ [°]')
axis[0, 0].grid(axis='y', which="major", linewidth=1)
axis[0, 0].grid(axis='y', which="minor", linewidth=0.2)
axis[0, 0].tick_params(axis='y', which='minor')
axis[0, 0].tick_params(axis='y', which='major')
markerline, stemlines, baseline = axis[0, 0].stem(X1, Y1)
plt.setp(markerline, 'color', acin_colors['TU_blue'])
plt.setp(stemlines, 'color', acin_colors['TU_blue'])
axis[0, 0].set_xticks([0,1,2,3,4,5,6,7])

axis[0, 1].clear()
Y2 = param_errors[6:13, -1] * 1000
Z2 = -d_error * 1000
X2 = np.arange(1, len(Y2)+1)
_min = np.concatenate((Y2, Z2)).min()
_max = np.concatenate((Y2, Z2)).max()
_abs = max(abs(_min), abs(_max))
_min = _min + np.sign(_min)*0.1*_abs  # add a buffer of 10% of the max value on top and on bottom
_max = _max + np.sign(_max)*0.1*_abs
axis[0, 1].set_ylim([_min, _max])
axis[0, 1].scatter(X, Z2, c=acin_colors['acin_yellow'], s=100)
axis[0, 1].set_title(r'$\Delta$d [mm]')
axis[0, 1].grid(axis='y', which="major", linewidth=1)
axis[0, 1].grid(axis='y', which="minor", linewidth=0.2)
axis[0, 1].tick_params(axis='y', which='minor')
axis[0, 1].tick_params(axis='y', which='major')
markerline, stemlines, baseline = axis[0, 1].stem(X2, Y2)
plt.setp(markerline, 'color', acin_colors['TU_blue'])
plt.setp(stemlines, 'color', acin_colors['TU_blue'])
axis[0, 1].set_xticks([0,1,2,3,4,5,6,7])

axis[1, 0].clear()
Y3 = param_errors[13:19, -1] * 1000
Z3 = -r_error * 1000
X3 = np.arange(1, len(Y3)+1)
_min = np.concatenate((Y3, Z3)).min()
_max = np.concatenate((Y3, Z3)).max()
_abs = max(abs(_min), abs(_max))
_min = _min + np.sign(_min)*0.1*_abs  # add a buffer of 10% of the max value on top and on bottom
_max = _max + np.sign(_max)*0.1*_abs
axis[1, 0].set_ylim([_min, _max])
axis[1, 0].scatter(X, Z3, c=acin_colors['acin_yellow'], s=100)
axis[1, 0].set_title(r'$\Delta$r [mm]')
axis[1, 0].grid(axis='y', which="major", linewidth=1)
axis[1, 0].grid(axis='y', which="minor", linewidth=0.2)
axis[1, 0].tick_params(axis='y', which='minor')
axis[1, 0].tick_params(axis='y', which='major')
markerline, stemlines, baseline = axis[1, 0].stem(X3, Y3)
plt.setp(markerline, 'color', acin_colors['TU_blue'])
plt.setp(stemlines, 'color', acin_colors['TU_blue'])
axis[1, 0].set_xticks([0,1,2,3,4,5,6,7])

axis[1, 1].clear()
Y4 = param_errors[19:26, -1] * 180 / np.pi
Z4 = -alpha_error * 180 / np.pi
X4 = np.arange(1, len(Y4)+1)
_min = np.concatenate((Y4, Z4)).min()
_max = np.concatenate((Y4, Z4)).max()
_abs = max(abs(_min), abs(_max))
_min = _min + np.sign(_min)*0.1*_abs  # add a buffer of 10% of the max value on top and on bottom
_max = _max + np.sign(_max)*0.1*_abs
axis[1, 1].set_ylim([_min, _max])
axis[1, 1].scatter(X, Z4, c=acin_colors['acin_yellow'], s=100)
axis[1, 1].set_title(r'$\Delta$$\alpha$ [°]')
axis[1, 1].grid(axis='y', which="major", linewidth=1)
axis[1, 1].grid(axis='y', which="minor", linewidth=0.2)
axis[1, 1].tick_params(axis='y', which='minor')
axis[1, 1].tick_params(axis='y', which='major')
markerline, stemlines, baseline = axis[1, 1].stem(X4, Y4)
plt.setp(markerline, 'color', acin_colors['TU_blue'])
plt.setp(stemlines, 'color', acin_colors['TU_blue'])
axis[1, 1].set_xticks([0,1,2,3,4,5,6,7])
plt.savefig("ex10_data.pdf", format="pdf", bbox_inches="tight")

# fig_curr_est2, ax_curr_est2 = plt.subplots(2, 2, figsize=(6, 6))
# fig_curr_est2.tight_layout(pad=2)
# n = 8
# X = [e for e in range(0, n)]
# axis = ax_curr_est2
# param_errors = x.reshape((len(x), 1))
#
# axis[0, 0].clear()
#
# Y1 = param_errors[0:6, -1] * 180 / np.pi
# Z1 = -theta_error * 180 / np.pi
# X1 = np.arange(1, len(Y1)+1)
# _min = np.concatenate((Y1, Z1)).min()
# _max = np.concatenate((Y1, Z1)).max()
# _abs = max(abs(_min), abs(_max))
# _min = _min + np.sign(_min)*0.1*_abs  # add a buffer of 10% of the max value on top and on bottom
# _max = _max + np.sign(_max)*0.1*_abs
# axis[0, 0].set_ylim([_min, _max])
# axis[0, 0].scatter(X, Z1, c=acin_colors['acin_yellow'], s=100)
# axis[0, 0].set_title(r'$\Delta$$\theta$ [°]')
# axis[0, 0].grid(axis='y', which="major", linewidth=1)
# axis[0, 0].grid(axis='y', which="minor", linewidth=0.2)
# axis[0, 0].tick_params(axis='y', which='minor')
# axis[0, 0].tick_params(axis='y', which='major')
# markerline, stemlines, baseline = axis[0, 0].stem(X1, Y1)
# plt.setp(markerline, 'color', acin_colors['TU_blue'])
# plt.setp(stemlines, 'color', acin_colors['TU_blue'])
# axis[0, 0].set_xticks([0,1,2,3,4,5,6,7])
#
# axis[0, 1].clear()
# Y2 = param_errors[6:13, -1] * 1000
# Z2 = -d_error * 1000
# X2 = np.arange(1, len(Y2)+1)
# _min = np.concatenate((Y2, Z2)).min()
# _max = np.concatenate((Y2, Z2)).max()
# _abs = max(abs(_min), abs(_max))
# _min = _min + np.sign(_min)*0.1*_abs  # add a buffer of 10% of the max value on top and on bottom
# _max = _max + np.sign(_max)*0.1*_abs
# axis[0, 1].set_ylim([_min, _max])
# axis[0, 1].scatter(X, Z2, c=acin_colors['acin_yellow'], s=100)
# axis[0, 1].set_title(r'$\Delta$d [mm]')
# axis[0, 1].grid(axis='y', which="major", linewidth=1)
# axis[0, 1].grid(axis='y', which="minor", linewidth=0.2)
# axis[0, 1].tick_params(axis='y', which='minor')
# axis[0, 1].tick_params(axis='y', which='major')
# markerline, stemlines, baseline = axis[0, 1].stem(X2, Y2)
# plt.setp(markerline, 'color', acin_colors['TU_blue'])
# plt.setp(stemlines, 'color', acin_colors['TU_blue'])
# axis[0, 1].set_xticks([0,1,2,3,4,5,6,7])
#
# axis[1, 0].clear()
# Y3 = param_errors[13:19, -1] * 1000
# Z3 = -r_error * 1000
# X3 = np.arange(1, len(Y3)+1)
# _min = np.concatenate((Y3, Z3)).min()
# _max = np.concatenate((Y3, Z3)).max()
# _abs = max(abs(_min), abs(_max))
# _min = _min + np.sign(_min)*0.1*_abs  # add a buffer of 10% of the max value on top and on bottom
# _max = _max + np.sign(_max)*0.1*_abs
# axis[1, 0].set_ylim([_min, _max])
# axis[1, 0].scatter(X, Z3, c=acin_colors['acin_yellow'], s=100)
# axis[1, 0].set_title(r'$\Delta$r [mm]')
# axis[1, 0].grid(axis='y', which="major", linewidth=1)
# axis[1, 0].grid(axis='y', which="minor", linewidth=0.2)
# axis[1, 0].tick_params(axis='y', which='minor')
# axis[1, 0].tick_params(axis='y', which='major')
# markerline, stemlines, baseline = axis[1, 0].stem(X3, Y3)
# plt.setp(markerline, 'color', acin_colors['TU_blue'])
# plt.setp(stemlines, 'color', acin_colors['TU_blue'])
# axis[1, 0].set_xticks([0,1,2,3,4,5,6,7])
#
# axis[1, 1].clear()
# Y4 = param_errors[19:26, -1] * 180 / np.pi
# Z4 = -alpha_error * 180 / np.pi
# X4 = np.arange(1, len(Y4)+1)
# _min = np.concatenate((Y4, Z4)).min()
# _max = np.concatenate((Y4, Z4)).max()
# _abs = max(abs(_min), abs(_max))
# _min = _min + np.sign(_min)*0.1*_abs  # add a buffer of 10% of the max value on top and on bottom
# _max = _max + np.sign(_max)*0.1*_abs
# axis[1, 1].set_ylim([_min, _max])
# axis[1, 1].scatter(X, Z4, c=acin_colors['acin_yellow'], s=100)
# axis[1, 1].set_title(r'$\Delta$$\alpha$ [°]')
# axis[1, 1].grid(axis='y', which="major", linewidth=1)
# axis[1, 1].grid(axis='y', which="minor", linewidth=0.2)
# axis[1, 1].tick_params(axis='y', which='minor')
# axis[1, 1].tick_params(axis='y', which='major')
# markerline, stemlines, baseline = axis[1, 1].stem(X4, Y4)
# plt.setp(markerline, 'color', acin_colors['TU_blue'])
# plt.setp(stemlines, 'color', acin_colors['TU_blue'])
# axis[1, 1].set_xticks([0,1,2,3,4,5,6,7])
# plt.savefig("ex10_data.pdf", format="pdf", bbox_inches="tight")

plt.show()




# df = pd.DataFrame()
# df['theta_error'] = (theta_error * 180 / np.pi).round(4)
# df['r_error'] = (r_error * 1000).round(4)
# df['d_error'] = (d_error * 1000).round(4)
# df['alpha_error'] = (alpha_error * 180 / np.pi).round(4)
# df['theta_error_m'] = (param_errors[0:n, -1] * 180 / np.pi).round(4)
# df['r_error_m'] = (param_errors[2*n:3*n, -1] * 1000).round(4)
# df['d_error_m'] = (param_errors[n:2*n, -1] * 1000).round(4)
# df['alpha_error_m'] = (param_errors[3*n:4*n, -1] * 180 / np.pi).round(4)
# df.to_csv('exp10_data.csv',
#           columns=['theta_error', 'theta_error_m', 'd_error', 'd_error_m', 'r_error', 'r_error_m', 'alpha_error', 'alpha_error_m'],
#           header=True,
#           index_label='id',
#           float_format="%.3f"
#           )

print("done")