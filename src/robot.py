import numpy as np
import utils
import cv2
from scipy.spatial.transform import Rotation as R
import math as m
from itertools import combinations
import random


class RobotDescription:
    """
    Defines the methods to calculate kinematics
    """
    pip2 = np.pi / 2
    pi = np.pi

    #nominal MDH Parameters of Kuka iiwa 14 with additional camera at ee GAZEBO
    dhparams = {"theta_nom": np.array([0.0, 0, 0, 0, 0, 0, 0, 0]),
               "d_nom": np.array([0.0, 0, 0, 0, 0, 0, 0, 0.1]),
                "r_nom": np.array([0, 0, 0.42, 0, 0.4, 0, 0.3, 0]),
                "alpha_nom": np.array([-pip2, pip2, -pip2, -pip2, pip2, pip2, -pip2, 1]),
                "num_joints": 7,
                "num_cam_extrinsic": 1}  # camera extrinsic calib


    assert dhparams['num_cam_extrinsic'] + dhparams['num_joints'] == len(dhparams['theta_nom']), ("Robot"
                          " MDH Configuration: stated number of joints does not match lenght of vector")

    # Correction matrix for camera between ros and opencv
    T_corr = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    # Transform from world frame to frame 0
    T_W0 = np.array([[-1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0.36],
                     [0, 0, 0, 1]])

    camera_params = {'camera_matrix': np.array([1386.4138492513919, 0.0, 960.5,
                                                    0.0, 1386.4138492513919, 540.5,
                                                    0.0, 0.0, 1.0]).reshape(3, 3),
                     'camera_distortion': np.zeros(5)}

    aruco_params = {'arucoDict': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000),
                    'aruco_length': 0.4,
                    'detector_params': cv2.aruco.DetectorParameters()}

    charuco_params = {'arucoDict': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
                      'detector_params': cv2.aruco.CharucoParameters(),
                      'refine_params': cv2.aruco.RefineParameters(),
                      'squares_verically': 7,
                      'squares_horizontally': 5,
                      'square_length': 0.03,
                      'marker_length': 0.015}



    def __init__(self):
        board_shape = (self.charuco_params['squares_verically'], self.charuco_params['squares_verically'])
        self.charuco_params['board'] = cv2.aruco.CharucoBoard(board_shape,
                                                              self.charuco_params['square_length'],
                                                              self.charuco_params['marker_length'],
                                                              self.charuco_params['arucoDict'])
        self.charuco_params['chdetector'] = cv2.aruco.CharucoDetector(self.charuco_params['board'],
                                                                      self.charuco_params['detector_params'],
                                                                      self.aruco_params['detector_params'],
                                                                      self.charuco_params['refine_params'])
    @staticmethod
    def get_linear_model(observation_pairs, theta, d, r, alpha, include_rot=False):
        """
        observation_pairs is a list of observation pairs for which the jacobian is to be computed
        """
        num_params = len(theta) + len(d) + len(r) + len(alpha)
        jacobian_tot = np.zeros((0, num_params))
        errors_tot = np.zeros(0)

        for obs1, obs2 in observation_pairs:
            pose_error = RobotDescription.get_model_error(obs1, obs2, theta, d, r, alpha)

            q1 = np.hstack((np.array(obs1["q"]), np.zeros(RobotDescription.dhparams['num_cam_extrinsic'])))
            q2 = np.hstack((np.array(obs2["q"]), np.zeros(RobotDescription.dhparams['num_cam_extrinsic'])))

            # calculate the corresponding difference jacobian
            jacobian = RobotDescription.get_parameter_jacobian_improved(q1=q1, q2=q2,
                                                          theta_all=theta,
                                                          d_all=d,
                                                          r_all=r,
                                                          alpha_all=alpha)

            if not include_rot:
                jacobian = jacobian[0:3, :]
                pose_error = pose_error[0:3]

            # collect the jacobian and error resulting from these two observations
            jacobian_tot = np.concatenate((jacobian_tot, jacobian), axis=0)
            errors_tot = np.concatenate((errors_tot, pose_error), axis=0)

        mat_q, mat_r = np.linalg.qr(jacobian_tot)
        diag_r = np.diagonal(mat_r)
        rank = np.linalg.matrix_rank(jacobian_tot)
        jacobian_quality = {'qr_diag_r_full_jacobian': diag_r, 'rank_full_jacobian': rank}

        # return jacobian_tot, errors_tot, list_marker_locations, jacobian_quality
        return jacobian_tot, errors_tot, jacobian_quality

    @staticmethod
    def get_parameter_jacobian_improved(q1, q2, theta_all, d_all, r_all, alpha_all) -> np.array:
        """
        Get the parameter jacobian, that is the matrix approximating the effect of parameter (DH)
        deviations on the final pose. The number of links is inferred from the length of the DH
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
        T_N1_0 = RobotDescription.get_T_jk(num_links, 0, q1, theta_all, d_all, r_all, alpha_all)  # T from N1 to 0
        T_0_N2 = RobotDescription.get_T_jk(0, num_links, q2, theta_all, d_all, r_all, alpha_all)  # T from 0 to N2
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
            T = T_N1_0 @ RobotDescription.get_T_jk(0, i+1, q2, theta_all, d_all, r_all, alpha_all)  # T from N1 to i2 (via 0)
            t = T[0:3, 3]
            R = T[0:3, 0:3]

            # compute vectors ui
            u_1 = np.array([m.cos(theta), - m.sin(theta), 0])
            u_2 = np.array([0, 0, 1])
            u_3 = np.array([- r * m.sin(theta), - r * m.cos(theta), 0])

            # compute vectors that make up columns of Jacobian
            j_1 = np.cross(t, (R @ u_2))
            j_2 = R @ u_1
            j_3 = R @ u_2
            j_4 = np.cross(t, (R @ u_1)) + R @ u_3
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
            T = RobotDescription.get_T_jk(num_links, i+1, q1, theta_all, d_all, r_all, alpha_all)  # T from N1 to i1
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


    @staticmethod
    def get_model_error(obs1, obs2, theta, d, r, alpha):
        """
        calculate the difference between calculated and measured pose difference of two observtations
        """

        q1 = np.hstack((np.array(obs1["q"]), np.zeros(RobotDescription.dhparams['num_cam_extrinsic'])))
        q2 = np.hstack((np.array(obs2["q"]), np.zeros(RobotDescription.dhparams['num_cam_extrinsic'])))

        T_CM_1 = obs1['mat']
        T_CM_2 = obs2['mat']


        T_0_cam_1 = RobotDescription.get_T_0_cam(q1, theta, d, r, alpha)
        T_0_cam_2 = RobotDescription.get_T_0_cam(q2, theta, d, r, alpha)

        # perform necessary inversions
        T_MC_2 = np.linalg.inv(T_CM_2)
        T_cam_0_1 = np.linalg.inv(T_0_cam_1)

        # T_WM_1 = pe.T_W0 @ T_08_1 @ T_CM_1
        # if markerid == 2 or True:
        #     list_marker_locations.append([T_WM_1[0, 3], T_WM_1[1, 3], T_WM_1[2, 3]])

        D_meas = T_CM_1 @ T_MC_2
        D_nom = T_cam_0_1 @ T_0_cam_2
        delta_D = D_meas @ np.linalg.inv(D_nom)
        delta_D_skew = 0.5 * (delta_D - delta_D.T)

        drvec = np.array([delta_D_skew[2, 1], delta_D_skew[0, 2], delta_D_skew[1, 0]])
        dtvec = delta_D[0:3, 3]
        pose_error = np.concatenate((dtvec, drvec))

        return pose_error


    @staticmethod
    def get_nominal_parameters():
        return RobotDescription.dhparams

    @staticmethod
    def observe(img, q, time=0):
        return RobotDescription.observe2(img, q, 'aruco', time)

    @staticmethod
    def observe2(img, q, marker_type, time=0):
        list_obs = []
        est_poses = RobotDescription.get_camera_tfs(img, marker_type)
        #joint_tfs = RobotDescription.get_joint_tfs(q)
        for pose in est_poses:
            obs = {"marker_id": pose['marker_id'],
                   "mat": pose['mat'],
                   "t": time,
                   "q": q}

            list_obs.append(obs)  # append observation to queue corresponding to id (deque from right)

        return list_obs

    @staticmethod
    def get_joint_tfs(q_vec):
        """
        from_frame : Name of the frame for which the transformation is added in the to_frame coordinate system
        to_frame :  Name of the frame in which the transformation is defined
        """
        joint_tfs = []  # initialize list
        params = RobotDescription.dhparams
        q_vec = q_vec.flatten()
        q_vec = np.append(q_vec, np.zeros(RobotDescription.dhparams["num_cam_extrinsic"]))  # pad q vector with zero for non actuated last transform
        for (i, q) in enumerate(q_vec):  # iterate over joint values
            theta = params["theta_nom"][i]
            d = params["d_nom"][i]
            r = params["r_nom"][i]
            alpha = params["alpha_nom"][i]
            joint_tfs.append({'mat': RobotDescription.get_T_i_forward(q, theta, d, r, alpha),
                              'from_frame': str(i+1), 'to_frame': str(i)})

        joint_tfs.append({'mat': RobotDescription.T_W0, 'from_frame': '0', 'to_frame': 'world'})
        return joint_tfs

    @staticmethod
    def get_camera_tfs(img, marker_type='aruco'):
        est_poses = []

        if marker_type == 'aruco':
            (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(img,
                                                                      RobotDescription.aruco_params['arucoDict'],
                                                                      parameters=RobotDescription.aruco_params['detector_params'])

            if marker_ids is None:
                return est_poses  # return empty list if no marker found

            # cv2.aruco.drawDetectedMarkers(img, corners, marker_ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                  RobotDescription.aruco_params['aruco_length'],
                                                                  RobotDescription.camera_params['camera_matrix'],
                                                                  RobotDescription.camera_params['camera_distortion'])

            # In OpenCV the pose of the marker is with respect to the camera lens frame.
            # Imagine you are looking through the camera viewfinder,
            # the camera lens frame's:
            # x-axis points to the right
            # y-axis points straight down towards your toes
            # z-axis points straight ahead away from your eye, out of the camera
        elif marker_type == 'charuco':
            res = RobotDescription.charuco_params['chdetector'].detectBoard(img)
            charucoCorners, charucoIds, chmarkerCorners, chmarkerIds = res
            rvec, tvec = np.zeros(3), np.zeros(3)
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners,
                                                                    charucoIds,
                                                                    RobotDescription.charuco_params['board'],
                                                                    RobotDescription.camera_params['camera_matrix'],
                                                                    RobotDescription.camera_params['camera_distortion'],
                                                                    rvec, tvec)
            if retval:
                marker_ids = [1]
            else:
                marker_ids = []

        for i, marker_id in enumerate(marker_ids.flatten()):
            T_ = utils.H_rvec_tvec(rvecs[i][0], tvecs[i][0])  # convert to homogeneous matrix

            # change base, as gazebo and cv use different camera directions:
            # opencv: x-axis points to the right
            #         y-axis points down
            #         z-axis points out of the camera
            # gazebo: x-axis points out of the camera
            #         y-axis points to the left
            #         z-axis points up
            T_CM = RobotDescription.T_corr @ T_ @ np.linalg.inv(RobotDescription.T_corr)
            est_poses.append({'mat': T_CM,
                              'marker_id': marker_id,
                              'from_frame': 'marker_'+str(marker_id),
                              'to_frame': 'camera'})
        return est_poses

    @staticmethod
    def get_alternate_tfs(tf_matrix):
        # Store the translation (i.e. position) information
        rotation_matrix, translation = utils.split_H_transform(tf_matrix)
        r = R.from_matrix(rotation_matrix[0:3, 0:3])
        return {'rotmat': rotation_matrix, 'quat': r.as_quat(), 'rvec': r.as_rotvec(), 'tvec': translation}


    @staticmethod
    def get_T_i_forward(q__i, theta__i, d__i, r__i, alpha__i, jointtype='revolute') -> np.array:
        Rx, Rz, Trans = utils.Rx, utils.Rz, utils.Trans
        if jointtype == 'revolute':
            # T = Rz(q__i+theta__i) @ Trans(0, 0, d__i) @ Trans(r__i, 0, 0) @ Rx(alpha__i)
            T = Rx(alpha__i) @ Trans(d__i, 0, 0) @ Rz(theta__i + q__i) @ Trans(0, 0, r__i)
        elif jointtype == 'prismatic':
            # T = Rz(theta__i) @ Trans(0, 0, q__i + d__i) @ Trans(r__i, 0, 0) @ Rx(alpha__i)
            T = Rx(alpha__i) @ Trans(d__i, 0, 0) @ Rz(theta__i) @ Trans(0, 0, r__i + q__i)
        else:
            return None
        return T

    @staticmethod
    def get_T_i_backward(q__i, theta__i, d__i, r__i, alpha__i, jointtype='revolute') -> np.array:
        Rx, Rz, Trans = utils.Rx, utils.Rz, utils.Trans
        if jointtype == 'revolute':
            # T = Rz(q__i+theta__i) @ Trans(0, 0, d__i) @ Trans(r__i, 0, 0) @ Rx(alpha__i)
            T = Trans(0, 0, - r__i) @ Rz(theta__i + q__i).T @ Trans(- d__i, 0, 0) @ Rx(alpha__i).T
        elif jointtype == 'prismatic':
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
        q = q.flatten()
        assert len(q) == len(theta_all) == len(d_all) == len(r_all) == len(alpha_all)
        theta_all, d_all, r_all, alpha_all = theta_all.flatten(), d_all.flatten(), r_all.flatten(), alpha_all.flatten()
        if j == k:  # transform is identity
            return T

        elif j > k:  # transform is in reverse direction, aka from k to j
            for i in range(k, j):
                _T = RobotDescription.get_T_i_forward(q[i], theta_all[i], d_all[i], r_all[i], alpha_all[i])
                T = T @ _T
            return np.linalg.inv(T)

        else:  # regular transform from j to k
            for i in range(j, k):
                _T = RobotDescription.get_T_i_forward(q[i], theta_all[i], d_all[i], r_all[i], alpha_all[i])
                T = T @ _T
            return T


    @staticmethod
    def get_T_0_cam(q, theta_all, d_all, r_all, alpha_all):
        j = 0
        k = RobotDescription.dhparams['num_joints'] + RobotDescription.dhparams['num_cam_extrinsic']
        return RobotDescription.get_T_jk(j, k, q, theta_all, d_all, r_all, alpha_all)

    @staticmethod
    def get_marker_location(T_CM, q, theta_all, d_all, r_all, alpha_all):
        T_0cam = RobotDescription.get_T_0_cam(q, theta_all, d_all, r_all, alpha_all)
        return RobotDescription.T_W0 @ T_0cam @ T_CM
