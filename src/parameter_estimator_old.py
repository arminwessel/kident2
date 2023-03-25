#!/usr/bin/env python3
import rospy
import numpy as np
import math as m
import cv2
from kident2.msg import DiffMeasurement, Array_f64
import utils


# from std_msgs.msg import Float32
# from std_srvs.srv import SetBool, SetBoolResponse
# import utils
class ParameterEstimator:
    """
    Estimate the kinematic model error of a robot manipulator
    The model is based on the DH convention
    """
    pip2 = np.pi / 2
    pi = np.pi
    dhparams = {"theta_nom": np.array([0.0, 0, 0, 0, 0, 0, 0]),
                "d_nom": np.array([0.0, 0, 0, 0, 0, 0, 0]),
                "r_nom": np.array([0, 0, 0.42, 0, 0.4, 0, 0]),
                "alpha_nom": np.array([0, pip2, -pip2, -pip2, pip2, pip2, -pip2])}

    # dhparams = {"theta_nom": np.array([0, pi, pi, 0, pi, 0, pi]),
    #             "d_nom" : np.array([0.1525, 0.2075, 0.2325, 0.1825, 0.2125, 0.1875, 0.081]),
    #             "r_nom" : np.array([0, 0, 0, 0, 0, 0, 0]),
    #             "alpha_nom" : np.array([0, pip2, pip2, pip2, pip2, pip2, pip2])}
    def __init__(self) -> None:
        """
        Constructor
        """
        self.sub_meas = rospy.Subscriber("diff_meas", DiffMeasurement, self.process_measurement)
        self.pub_est = rospy.Publisher("est", Array_f64, queue_size=20)

        self.theta_nom = ParameterEstimator.dhparams["theta_nom"]
        self.d_nom = ParameterEstimator.dhparams["d_nom"]
        self.r_nom = ParameterEstimator.dhparams["r_nom"]
        self.alpha_nom = ParameterEstimator.dhparams["alpha_nom"]

        num_links = self.theta_nom.size
        self.rls = RLS(4 * num_links, 1)
        self.distances = np.zeros((0,))

    @staticmethod
    def get_T_i_forward(q__i, theta__i, d__i, r__i, alpha__i, type='revolute') -> np.array:
        Rx, Rz, Trans = utils.Rx, utils.Rz, utils.Trans
        if type == 'revolute':
            # T = Rz(q__i+theta__i) @ Trans(0, 0, d__i) @ Trans(r__i, 0, 0) @ Rx(alpha__i)
            T = Rx(alpha__i) @ Trans(d__i, 0, 0) @ Rz(theta__i + q__i) @ Trans(0, 0, r__i)
        elif type == 'prismatic':
            # T = Rz(theta__i) @ Trans(0, 0, q__i + d__i) @ Trans(r__i, 0, 0) @ Rx(alpha__i)
            T = Rx(alpha__i) @ Trans(d__i, 0, 0) @ Rz(theta__i) @ Trans(0, 0, r__i + q__i)
        else:
            return None
        return T

    @staticmethod
    def get_T_i_backward(q__i, theta__i, d__i, r__i, alpha__i, type='revolute') -> np.array:
        Rx, Rz, Trans = utils.Rx, utils.Rz, utils.Trans
        if type == 'revolute':
            # T = Rz(q__i+theta__i) @ Trans(0, 0, d__i) @ Trans(r__i, 0, 0) @ Rx(alpha__i)
            T = Trans(0, 0, - r__i) @ Rz(theta__i + q__i).T @ Trans(- d__i, 0, 0) @ Rx(alpha__i).T
        elif type == 'prismatic':
            # T = Rz(theta__i) @ Trans(0, 0, q__i + d__i) @ Trans(r__i, 0, 0) @ Rx(alpha__i)
            T = Trans(0, 0, - r__i - q__i) @ Rz(theta__i).T @ Trans(- d__i, 0, 0) @ Rx(alpha__i).T
        else:
            return None
        return T

    @staticmethod
    def get_T_jk(j, k, q, theta_all, d_all, r_all, alpha_all) -> np.array:
        """
        T_jk = T_j^k
        """
        T = np.eye(4)
        q, theta_all, d_all, r_all, alpha_all = q.flatten(), theta_all.flatten(), d_all.flatten(), r_all.flatten(), alpha_all.flatten()
        if j == k:  # transform is identity
            return T

        elif j > k:  # transform is in reverse direction, aka from k to j
            for i in range(k, j):
                _T = ParameterEstimator.get_T_i_forward(q[i], theta_all[i], d_all[i], r_all[i], alpha_all[i])
                T = T @ _T
            return np.linalg.inv(T)

        else:  # regular transform from j to k
            for i in range(j, k):
                _T = ParameterEstimator.get_T_i_forward(q[i], theta_all[i], d_all[i], r_all[i], alpha_all[i])
                T = T @ _T
            return T


    def get_parameter_jacobian(self, q, theta_all, d_all, r_all, alpha_all) -> np.array:
        """
        Get the parameter jacobian, that is the matrix approximating the effect of parameter (DH)
        deviations on the final pose. The number of links is inferred from the lenght of the DH 
        parameter vectors. All joints are assumed rotational.
        """
        assert theta_all.size == d_all.size == r_all.size == alpha_all.size, "All parameter vectors must have same length"
        num_links = theta_all.size

        J1 = np.zeros((3, num_links))
        J2 = np.zeros((3, num_links))
        J3 = np.zeros((3, num_links))
        J4 = np.zeros((3, num_links))
        J5 = np.zeros((3, num_links))
        J6 = np.zeros((3, num_links))

        T_0N = self.get_T_jk(0, num_links, q, theta_all, d_all, r_all, alpha_all)
        t_0N = T_0N[0:3, 3]
        for i in range(num_links):   # iterate over the links of the robot
            theta = theta_all[i]
            d = d_all[i]
            r = r_all[i]
            alpha = alpha_all[i]

            T_0i = self.get_T_jk(0, i, q, theta_all, d_all, r_all, alpha_all)  # get the transformation to link i
            t_0i = T_0i[0:3, 3]
            R_0i = T_0i[0:3, 0:3]

            # compute vectors vi
            v_1i = np.array([0, - d * m.cos(alpha), - d * m.sin(alpha)])
            v_2i = np.array([1, 0, 0])
            v_3i = np.array([0, - m.sin(alpha), m.cos(alpha)])

            # compute vectors that make up columns of Jacobian
            J1_i = R_0i @ v_1i + np.cross(t_0i, (R_0i @ v_3i))
            J2_i = R_0i @ v_2i
            J3_i = R_0i @ v_3i
            J4_i = np.cross(t_0i, (R_0i @ v_2i))
            J5_i = np.cross(J3_i, t_0N) + J1_i
            J6_i = np.cross(J2_i, t_0N) + J4_i

            # add vectors as columns
            J1[:, i] = J1_i
            J2[:, i] = J2_i
            J3[:, i] = J3_i
            J4[:, i] = J4_i
            J5[:, i] = J5_i
            J6[:, i] = J6_i

        J = np.zeros((6, 4 * num_links))
        J0 = np.zeros((3, num_links))
        J[0:3, :] = np.concatenate((J5, J2, J3, J6), axis=1)  # upper part of Jacobian is for differential translation
        J[3:6, :] = np.concatenate((J3, J0, J0, J2), axis=1)  # lower part is for differential rotation
        return J

    def get_parameter_jacobian_dual(self, q1, q2, theta_all, d_all, r_all, alpha_all) -> np.array:
        """
        Get the parameter jacobian, that is the matrix approximating the effect of parameter (DH)
        deviations on the final pose. The number of links is inferred from the lenght of the DH
        parameter vectors. All joints are assumed rotational.
        """
        assert theta_all.size == d_all.size == r_all.size == alpha_all.size, "All parameter vectors must have same length"
        num_links = theta_all.size

        J1 = np.zeros((3, num_links))
        J2 = np.zeros((3, num_links))
        J3 = np.zeros((3, num_links))
        J4 = np.zeros((3, num_links))
        J5 = np.zeros((3, num_links))
        J6 = np.zeros((3, num_links))

        # calculate the backwards chain with q1
        T_N10 = self.get_T_jk(num_links, 0, q1, theta_all, d_all, r_all, alpha_all)  # T from N1 to 0
        t_N10 = T_N10[0:3, 3]
        for i in reversed(range(num_links)):   # iterate over the links backwards
            theta = theta_all[i]
            d = d_all[i]
            r = r_all[i]
            alpha = alpha_all[i]

            # get the transformation to link i using q1
            # the first transform is N to N, which is identity
            # the last transform is to bring error in frame 1 to frame N
            T_N1i = self.get_T_jk(num_links, i+1, q1, theta_all, d_all, r_all, alpha_all)  # T from N1 to i
            t_N1i = T_N1i[0:3, 3]
            R_N1i = T_N1i[0:3, 0:3]

            # compute vectors wi
            w_1i = np.array([- m.cos(theta), m.sin(theta), 0])
            w_2i = np.array([0, 0, -1])
            w_3i = np.array([r * m.sin(theta), r * m.cos(theta), 0])

            # compute vectors that make up columns of Jacobian
            j_i1 = np.cross(t_N1i, (R_N1i @ w_2i))
            j_i2 = R_N1i @ w_1i
            j_i3 = R_N1i @ w_2i
            j_i4 = np.cross(t_N1i, (R_N1i @ w_1i)) + R_N1i @ w_3i
            j_i5 = np.cross(j_i3, t_N10) + j_i1
            j_i6 = np.cross(j_i2, t_N10) + j_i4

            # add vectors to columns
            J1[:, i] += j_i1
            J2[:, i] += j_i2
            J3[:, i] += j_i3
            J4[:, i] += j_i4
            J5[:, i] += j_i5
            J6[:, i] += j_i6

        # calculate the forwards chain with q2
        T_0N2 = self.get_T_jk(0, num_links, q2, theta_all, d_all, r_all, alpha_all)  # T from 0 to N2
        T_N1N2 = T_N10 @ T_0N2
        t_N1N2 = T_N1N2[0:3, 3]
        for i in range(num_links):   # iterate over the links of the robot
            theta = theta_all[i]
            d = d_all[i]
            r = r_all[i]
            alpha = alpha_all[i]

            # get the transformation to link i using q1
            T_0i = self.get_T_jk(0, i, q2, theta_all, d_all, r_all, alpha_all)
            T_N1i = T_N10 @ T_0i
            t_N1i = T_N1i[0:3, 3]
            R_N1i = T_N1i[0:3, 0:3]

            # compute vectors vi
            v_1i = np.array([0, - d * m.cos(alpha), - d * m.sin(alpha)])
            v_2i = np.array([1, 0, 0])
            v_3i = np.array([0, - m.sin(alpha), m.cos(alpha)])

            # compute vectors that make up columns of Jacobian
            j_1 = R_N1i @ v_1i + np.cross(t_N1i, (R_N1i @ v_3i))
            j_2 = R_N1i @ v_2i
            j_3 = R_N1i @ v_3i
            j_4 = np.cross(t_N1i, (R_N1i @ w_1i)) + R_N1i @ w_3i
            j_5 = np.cross(j_3, t_N1N2) + j_1
            j_6 = np.cross(j_2, t_N1N2) + j_4

            # add vectors to columns
            J1[:, i] += j_1
            J2[:, i] += j_2
            J3[:, i] += j_3
            J4[:, i] += j_4
            J5[:, i] += j_5
            J6[:, i] += j_6

        J = np.zeros((6, 4 * num_links))
        J0 = np.zeros((3, num_links))
        J[0:3, :] = np.concatenate((J5, J2, J3, J6), axis=1)  # upper part of Jacobian is for differential translation
        J[3:6, :] = np.concatenate((J3, J0, J0, J2), axis=1)  # lower part is for differential rotation
        return J

    def get_parameter_jacobian_dual_2(self, q1, q2, theta_all, d_all, r_all, alpha_all) -> np.array:
        """
        Get the parameter jacobian, that is the matrix approximating the effect of parameter (DH)
        deviations on the final pose. The number of links is inferred from the lenght of the DH
        parameter vectors. All joints are assumed rotational.
        """
        assert theta_all.size == d_all.size == r_all.size == alpha_all.size, "All parameter vectors must have same length"
        num_links = theta_all.size

        J1 = np.zeros((3, num_links))
        J2 = np.zeros((3, num_links))
        J3 = np.zeros((3, num_links))
        J4 = np.zeros((3, num_links))
        J5 = np.zeros((3, num_links))
        J6 = np.zeros((3, num_links))

        # Total chain
        T_N1_0 = self.get_T_jk(num_links, 0, q1, theta_all, d_all, r_all, alpha_all)  # T from N1 to 0
        T_0_N2 = self.get_T_jk(0, num_links, q2, theta_all, d_all, r_all, alpha_all)  # T from 0 to N2
        T_tot = T_N1_0 @ T_0_N2
        t_tot = T_tot[0:3, 3]


        for i in range(num_links):  # iterate over the links of the robot (0, 1, ..., num_links-1)
        # calculate the forwards chain
            # parameters for current link
            theta = theta_all[i] + q2[i]
            d = d_all[i]
            r = r_all[i]
            alpha = alpha_all[i]
            # coordinate transform for current link
            T = T_N1_0 @ self.get_T_jk(0, i, q2, theta_all, d_all, r_all, alpha_all)  # T from N1 to i2 (via 0)
            t = T[0:3, 3]
            R = T[0:3, 0:3]

            # compute vectors vi
            v_1 = np.array([0, - d * m.cos(alpha), - d * m.sin(alpha)])
            v_2 = np.array([1, 0, 0])
            v_3 = np.array([0, - m.sin(alpha), m.cos(alpha)])

            # compute vectors that make up columns of Jacobian
            j_1 = R @ v_1 + np.cross(t, (R @ v_3))
            j_2 = R @ v_2
            j_3 = R @ v_3
            j_4 = np.cross(t, (R @ v_2))
            j_5 = np.cross(j_3, t_tot) + j_1
            j_6 = np.cross(j_2, t_tot) + j_4

            # add vectors to columns
            J1[:, i] += j_1
            J2[:, i] += j_2
            J3[:, i] += j_3
            J4[:, i] += j_4
            J5[:, i] += j_5
            J6[:, i] += j_6

        # calculate the reverse chain
            # parameters for current link
            theta = theta_all[i] + q1[i]
            d = d_all[i]
            r = r_all[i]
            alpha = alpha_all[i]

            # coordinate transform for current link
            T = self.get_T_jk(num_links, i+1, q1, theta_all, d_all, r_all, alpha_all)  # T from N1 to i1
            t = T[0:3, 3]
            R = T[0:3, 0:3]

            # compute vectors wi
            w_1 = np.array([- m.cos(theta), m.sin(theta), 0])
            w_2 = np.array([0, 0, -1])
            w_3 = np.array([r * m.sin(theta), r * m.cos(theta), 0])

            # compute vectors that make up columns of Jacobian
            j_1 = np.cross(t, (R @ w_2))
            j_2 = R @ w_1
            j_3 = R @ w_2
            j_4 = np.cross(t, (R @ w_1)) + R @ w_3
            j_5 = np.cross(j_3, t_tot) + j_1
            j_6 = np.cross(j_2, t_tot) + j_4

            # add vectors to columns
            J1[:, i] += j_1
            J2[:, i] += j_2
            J3[:, i] += j_3
            J4[:, i] += j_4
            J5[:, i] += j_5
            J6[:, i] += j_6

        J = np.zeros((6, 4 * num_links))
        J0 = np.zeros((3, num_links))
        J[0:3, :] = np.concatenate((J5, J2, J3, J6), axis=1)  # upper part of Jacobian is for differential translation
        J[3:6, :] = np.concatenate((J3, J0, J0, J2), axis=1)  # lower part is for differential rotation
        return J

    def get_parameter_jacobian_dual_3(self, q1, q2, theta_all, d_all, r_all, alpha_all) -> np.array:
        """
        Get the parameter jacobian, that is the matrix approximating the effect of parameter (DH)
        deviations on the final pose. The number of links is inferred from the lenght of the DH
        parameter vectors. All joints are assumed rotational.
        """
        assert theta_all.size == d_all.size == r_all.size == alpha_all.size, "All parameter vectors must have same length"
        num_links = theta_all.size

        J1 = np.zeros((3, num_links))
        J2 = np.zeros((3, num_links))
        J3 = np.zeros((3, num_links))
        J4 = np.zeros((3, num_links))
        J5 = np.zeros((3, num_links))
        J6 = np.zeros((3, num_links))

        # Total chain
        T_N1_0 = self.get_T_jk(num_links, 0, q1, theta_all, d_all, r_all, alpha_all)  # T from N1 to 0
        T_0_N2 = self.get_T_jk(0, num_links, q2, theta_all, d_all, r_all, alpha_all)  # T from 0 to N2
        T_tot = T_N1_0 @ T_0_N2  # corresponds to the nominal movement of the camera
        t_tot = T_tot[0:3, 3]

        for i in range(num_links):  # iterate over the links of the robot (0, 1, ..., num_links-1)
            # calculate nominal transform from N1 (camera pose for q1) to the frame of the current link for q1
            # T_N1_i1 = self.get_T_jk(num_links, i+1, q1, theta_all, d_all, r_all, alpha_all)  # T from N1 to i1
            # calculate the forwards chain
            # parameters for current link
            theta = theta_all[i] + q2[i]
            d = d_all[i]
            r = r_all[i]
            alpha = alpha_all[i]
            # coordinate transform for current link
            T = T_N1_0 @ self.get_T_jk(0, i + 1, q2, theta_all, d_all, r_all, alpha_all)  # T from N1 to i2 (via 0)
            t = T[0:3, 3]
            R = T[0:3, 0:3]

            # compute vectors vi
            v_1 = np.array([0, - d * m.cos(alpha), - d * m.sin(alpha)])
            v_2 = np.array([1, 0, 0])
            v_3 = np.array([0, - m.sin(alpha), m.cos(alpha)])

            # compute vectors that make up columns of Jacobian
            j_1 = R @ v_1 + np.cross(t, (R @ v_3))
            j_2 = R @ v_2
            j_3 = R @ v_3
            j_4 = np.cross(t, (R @ v_2))
            j_5 = np.cross(j_3, t_tot) + j_1
            j_6 = np.cross(j_2, t_tot) + j_4

            # add vectors to columns
            J1[:, i] += j_1
            J2[:, i] += j_2
            J3[:, i] += j_3
            J4[:, i] += j_4
            J5[:, i] += j_5
            J6[:, i] += j_6

            # calculate the reverse chain
            # parameters for current link
            theta = theta_all[i] + q1[i]
            d = d_all[i]
            r = r_all[i]
            alpha = alpha_all[i]

            # coordinate transform for current link
            T = self.get_T_jk(num_links, i + 1, q1, theta_all, d_all, r_all, alpha_all)  # T from N1 to i1
            t = T[0:3, 3]
            R = T[0:3, 0:3]

            # compute vectors wi
            w_1 = np.array([0, d * m.cos(alpha), d * m.sin(alpha)])
            w_2 = np.array([- 1, 0, 0])
            w_3 = np.array([0, m.sin(alpha), - m.cos(alpha)])

            # compute vectors that make up columns of Jacobian
            j_1 = np.cross(t, (R @ w_2))
            j_2 = R @ w_1
            j_3 = R @ w_2
            j_4 = np.cross(t, (R @ w_1)) + R @ w_3
            j_5 = np.cross(j_3, t_tot) + j_1
            j_6 = np.cross(j_2, t_tot) + j_4

            # add vectors to columns
            J1[:, i] += j_1
            J2[:, i] += j_2
            J3[:, i] += j_3
            J4[:, i] += j_4
            J5[:, i] += j_5
            J6[:, i] += j_6

        J = np.zeros((6, 4 * num_links))
        J0 = np.zeros((3, num_links))
        J[0:3, :] = np.concatenate((J5, J2, J3, J6), axis=1)  # upper part of Jacobian is for differential translation
        J[3:6, :] = np.concatenate((J3, J0, J0, J2), axis=1)  # lower part is for differential rotation
        return J

    def get_parameter_jacobian_improved(self, qA, qB, theta_all, d_all, r_all, alpha_all) -> np.array:
        """
        Get the parameter jacobian, that is the matrix approximating the effect of parameter (DH)
        deviations on the final pose. The number of links is inferred from the lenght of the DH
        parameter vectors. All joints are assumed rotational.
        """
        assert theta_all.size == d_all.size == r_all.size == alpha_all.size, "All parameter vectors must have same length"
        num_links = theta_all.size

        J = np.zeros((6, 4*num_links))

        for i in range(num_links):  # iterate over the links of the robot (0, 1, ..., num_links-1)
            # calculate total nominal
            T_tot = self.get_T_jk(num_links, 0, qA, theta_all, d_all, r_all, alpha_all) @ self.get_T_jk(0, num_links, qB, theta_all, d_all, r_all, alpha_all)
            R_tot = T_tot[0:3, 0:3]
            p_tot = T_tot[0:3, 3]

            # parameters for current link
            theta = theta_all[i] + qA[i]
            d = d_all[i]
            r = r_all[i]
            alpha = alpha_all[i]

            # compute vectors wi
            w_1 = np.array([- m.cos(theta), m.sin(theta), 0]) # for d and alpha
            w_2 = np.array([0, 0, -1]) # for r and theta
            w_3 = np.array([r * m.sin(theta), r * m.cos(theta), 0]) # for alpha

            # parameters for current link
            theta = theta_all[i] + qB[i]
            d = d_all[i]
            r = r_all[i]
            alpha = alpha_all[i]

            # compute vectors vi
            v_1 = np.array([0, - d * m.cos(alpha), - d * m.sin(alpha)])
            v_2 = np.array([1, 0, 0])
            v_3 = np.array([0, - m.sin(alpha), m.cos(alpha)])

            # Transform B errors to A
            T = self.get_T_jk(i+1, 0, qA, theta_all, d_all, r_all, alpha_all) @ self.get_T_jk(0, i+1, qB, theta_all, d_all, r_all, alpha_all)
            R = T[0:3, 0:3]
            p = T[0:3, 3]

            u1 = R @ v_1 + np.cross(p, R @ v_3)
            u2 = R @ v_2
            u3 = R @ v_3
            u4 = np.cross(p, (R @ v_2))
            u5 = R @ v_3
            u6 = R @ v_2

            # Add up errors in A
            x1 = u1
            x2 = u2 + w_1
            x3 = u3 + w_2
            x4 = u4 + w_3
            x5 = u5 + w_2
            x6 = u6 + w_1

            # coordinate transform from last link in config A to current link
            T = self.get_T_jk(num_links, i + 1, qA, theta_all, d_all, r_all, alpha_all)
            R = T[0:3, 0:3]
            p = T[0:3, 3]

            # This will now be transformed to the last link of config A
            j1 = (R @ x1) + np.cross(p, (R @ x5)) + np.cross((R @ x5), p_tot)
            j2 = R @ x2
            j3 = R @ x3
            j4 = (R @ x4) + np.cross(p, (R @ x6)) + np.cross((R @ x6), p_tot)
            j5 = R @ x5
            j6 = R @ x6

            # add results to Jacobian
            J[0:3, i] = j1
            J[0:3, i+num_links] = j2
            J[0:3, i + 2*num_links] = j3
            J[0:3, i + 3*num_links] = j4
            J[3:6, i] = j5
            J[3:6, i + 3 * num_links] = j6


        return J

    def process_measurement(self, m):
        # calculate the poses of the camera based on the nominal 
        # parameters and forward kinematics
        num_links = (np.array(m.q1)).size

        # compute rvec and tvec of kinematic chain with nominal values for coordinates q1
        q1 = np.array(m.q1)
        T_0N_1 = self.get_T_jk(0, num_links, q1, self.theta_nom, self.d_nom, self.r_nom, self.alpha_nom)
        tvec_0N_1 = T_0N_1[0:3, 3].reshape((3, 1))
        rvec_0N_1 = cv2.Rodrigues(T_0N_1[0:3, 0:3])[0]

        # compute rvec and tvec of kinematic chain with nominal values for coordinates q2
        q2 = np.array(m.q2)
        T_0N_2 = self.get_T_jk(0, num_links, q2, self.theta_nom, self.d_nom, self.r_nom, self.alpha_nom)
        tvec_0N_2 = T_0N_2[0:3, 3].reshape((3, 1))
        rvec_0N_2 = cv2.Rodrigues(T_0N_2[0:3, 0:3])[0]

        # calculate the difference in the nominal poses
        dtvec_nom = tvec_0N_1 - tvec_0N_2
        drvec_nom = rvec_0N_1 - rvec_0N_2

        # calculate the error between th expected and measured pose differences
        dtvec_real, drvec_real = np.reshape(np.array(m.dtvec), (3, 1)), np.reshape(np.array(m.drvec), (3, 1))
        current_error = np.concatenate((dtvec_real - dtvec_nom, drvec_real - drvec_nom), axis=0)

        # calculate the corresponding difference jacobian
        jacobian1 = self.get_parameter_jacobian(q1, self.theta_nom, self.d_nom, self.r_nom, self.alpha_nom)
        jacobian2 = self.get_parameter_jacobian(q2, self.theta_nom, self.d_nom, self.r_nom, self.alpha_nom)
        jacobian = jacobian1 - jacobian2

        # use RLS
        self.rls.add_obs(S=jacobian, Y=current_error)
        estimate_k = self.rls.get_estimate().flatten()

        # compose and publish message containing estimate
        msg = Array_f64()
        msg.data = estimate_k
        msg.time = rospy.get_time()
        self.pub_est.publish(msg)


class RLS:
    def __init__(self, num_params, q, alpha=1e3) -> None:
        """
        num_params: number of parameters to be estimated
        q: forgetting factor, usually very close to 1.
        alpha: initial value on diagonal of P
        """
        assert 1 >= q > 0.95, "q usually needs to be from ]0.95, 1]"
        self.q = q

        self.alpha = alpha
        self.num_params = num_params
        self.P = alpha * np.eye(num_params)  # initial value of matrix P
        self.phat = np.zeros((num_params, 1))  # initial guess for parameters, col vector
        self.num_obs = 0

    def add_obs(self, S, Y) -> None:
        """
        Add an observation
        S_T: array of data vectors [[s1],[s2],[s3]...]
        Y: measured outputs vector [[y1],[y2],[y3]...]
        """
        if S.ndim == 1:  # 1D arrays are converted to a row in a 2D array
            S = np.reshape(S, (1, -1))
        if Y.ndim == 1:
            Y = np.reshape(Y, (-1, 1))

        assert np.shape(S)[1] == self.num_params, "number of parameters has to agree with measurement dim"
        assert np.shape(S)[0] == np.shape(Y)[0], "observation dimensions don't match"

        for obs in zip(S, Y):  # iterate over rows, each iteration is an independent measurement
            (s_T, y) = obs
            s_T = np.reshape(s_T, (1, -1))
            s = np.transpose(s_T)
            _num = self.P @ s
            _den = (self.q + s_T @ self.P @ s)
            self.k = _num / _den

            self.P = (self.P - self.k @ s_T @ self.P) * (1 / self.q)

            self.phat = self.phat + self.k * (y - s_T @ self.phat)
            self.num_obs = self.num_obs + 1

        # if (np.any(np.abs(self.phat)>0.15)): # values are too big, reset LSQ
        #     self.P = self.alpha*np.eye(self.num_params) #initial value of matrix P
        #     self.phat = np.zeros((self.num_params,1)) #initial guess for parameters, col vector

    def get_estimate(self):
        return self.phat

    def get_num_obs(self):
        return self.num_obs


# Main function.
if __name__ == "__main__":
    rospy.init_node('dh_estimator')  # init ROS node
    rospy.loginfo('#Node dh_estimator running#')

    while not rospy.get_rostime():  # wait for ros time service
        pass
    pe = ParameterEstimator()  # create instance
    rospy.spin()
