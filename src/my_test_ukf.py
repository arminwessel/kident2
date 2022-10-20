import numpy as np
from parameter_estimator import ParameterEstimator
import utils
from scipy import linalg
from Pose_Estimation_Class import UKF
from helpers import Tools

theta_nom = ParameterEstimator.dhparams["theta_nom"]
r_nom = ParameterEstimator.dhparams["r_nom"]
d_nom = ParameterEstimator.dhparams["d_nom"]
alpha_nom = ParameterEstimator.dhparams["alpha_nom"]

ukf = UKF()

# X = np.array([[ 0.8660254,  0.0000000,  0.5000000, 0.0000000],
#               [ 0.0000000,  1.0000000,  0.0000000, 0.0000000],
#               [-0.5000000,  0.0000000,  0.8660254, 0.3000000],
#               [ 0.0000000,  0.0000000,  0.0000000, 1.0000000]])
X = np.zeros((4, 4))
X[0:3, 0:3] = Tools.vec2rotmat(np.pi/4, np.array([1, 0, 0]))
X[0:3,3] = np.array([0.12,0.13,0.5])
X[3,3] = 1
Xinv = np.linalg.inv(X)

U = np.array([[-1.0000000,  0.0000000,  0.0000000, 2.0000000],
              [ 0.0000000, -1.0000000,  0.0000000, 2.0000000],
              [ 0.0000000,  0.0000000,  1.0000000, 1.0000000],
              [ 0.0000000,  0.0000000,  0.0000000, 1.0000000]])

q = (np.random.rand(7) - np.full((7,), 0.5)) * np.pi / 2
avg = 0
for i in range(1000):
    q = q + (np.random.rand(7) - np.full((7,), 0.5)) * np.pi / 2
    avg = avg + np.mean(np.abs(q))
    T1 = ParameterEstimator.get_T__i0(7, q, d_nom, r_nom, alpha_nom)
    M1 = Xinv @ np.linalg.inv(T1) @ U

    q = q + (np.random.rand(7) - np.full((7,), 0.5)) * np.pi / 2
    avg = avg + np.mean(np.abs(q))
    T2 = ParameterEstimator.get_T__i0(7, q, d_nom, r_nom, alpha_nom)
    M2 = Xinv @ np.linalg.inv(T2) @ U


    BB = M1 @ np.linalg.inv(M2)
    AA = np.linalg.inv(T1) @ T2

    # test=AA@Xinv@np.linalg.inv(BB)@X
    # test = X@AA@Xinv@np.linalg.inv(BB)
    # utils.roundprint(test)

    ukf.Update(AA, BB)
avg = avg/1000
print("avg : {}".format(avg))


theta = np.linalg.norm(ukf.x[:3])
EPS = 0.00001
if theta < EPS:
    k = [0, 1, 0]  # VRML standard
else:
    k = ukf.x[0:3] / np.linalg.norm(ukf.x[:3])
euler_ukf = Tools.mat2euler(Tools.vec2rotmat(theta, k))
print('\n')
print('.....UKF Results')
euler_GT = Tools.mat2euler(X[:3, :3])
ukf_euler_err = np.array(euler_ukf) * 180 / np.pi - np.array(euler_GT) * 180 / np.pi
ukf_pos_err = ukf.x[3:].T * 100 - X[:3, 3].T * 100
print("GT[euler_rpy(deg) , pos(mm)]:", np.array(euler_GT) * 180 / np.pi, X[:3, 3].T * 100)
print("UKF [euler_rpy(deg) , pos(mm)]:", np.array([euler_ukf]) * 180 / np.pi, ukf.x[3:] * 100)
print("Error[euler_rpy(deg) , pos(mm)]:", ukf_euler_err, ukf_pos_err)
import matplotlib.pyplot as plt
plt.plot(ukf.consistency)
plt.show()