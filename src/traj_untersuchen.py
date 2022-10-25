import pandas as pd
import numpy as np
from parameter_estimator import ParameterEstimator

theta_nom = ParameterEstimator.dhparams["theta_nom"]
r_nom = ParameterEstimator.dhparams["r_nom"]
d_nom = ParameterEstimator.dhparams["d_nom"]
alpha_nom = ParameterEstimator.dhparams["alpha_nom"]

traj_str = "/home/armin/catkin_ws/src/kident2/src/traj.csv"
try:
    df = pd.read_csv(traj_str)
except:
    print("Could not load trajectory from {}".format(traj_str))

traj = df.to_numpy()  # shape: (num_joints, num_traj_points)
traj = traj[:, 1:]  # delete header
# move certain joints only
# traj[0, :] = np.zeros(5000)
# traj[1, :] = np.zeros(5000)
# traj[2, :] = np.zeros(5000)
# traj[3, :] = np.zeros(5000)
# traj[4, :] = np.zeros(5000)
# traj[5, :] = np.zeros(5000)
# traj[6, :] = np.zeros(5000)

traj = np.zeros((7, 50))
traj[0, :] = np.random.uniform(low=-170, high=170, size=(50,))
traj[1, :] = np.random.uniform(low=-120, high=120, size=(50,))
traj[2, :] = np.random.uniform(low=-170, high=170, size=(50,))
traj[3, :] = np.random.uniform(low=-120, high=120, size=(50,))
traj[4, :] = np.random.uniform(low=-170, high=170, size=(50,))
traj[5, :] = np.random.uniform(low=-120, high=120, size=(50,))
traj[6, :] = np.random.uniform(low=-175, high=175, size=(50,))

traj = traj*np.pi/180

df = pd.DataFrame(traj)
df.to_csv("traj.csv")
ee_poses = np.zeros((6, 5000))

for (idq, q) in enumerate(np.transpose(traj)):
    T = ParameterEstimator.get_T__i0(7, q, d_nom, r_nom, alpha_nom)
    trans_z = T@np.array([0,0,1,1])
    ee_poses[:, idq] = np.concatenate((T[0:3, 3].flatten(), trans_z[0:3].flatten()))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Quiver plot")

#drawing quiver plot
X, Y, Z, U, V, W = ee_poses[0, :], ee_poses[1, :], ee_poses[2, :], ee_poses[3, :], ee_poses[4, :], ee_poses[5, :]

ax.scatter(X, Y, Z, c='green')
ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)

# Create a sphere
r = 0.82
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
sphere_x = r*sin(phi)*cos(theta)
sphere_y = r*sin(phi)*sin(theta)
sphere_z = r*cos(phi)


ax.plot_surface(
    sphere_x, sphere_y, sphere_z,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0.3, edgecolor = 'k')

plt.show()
