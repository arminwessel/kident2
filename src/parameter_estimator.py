#!/usr/bin/env python3
import rospy
import numpy as np
import math as m
import cv2
from kident2.msg import DiffMeasurement,Array_f64
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
    dhparams = {"theta_nom": np.array([0, 0, 0, 0, 0, 0, 0]),
                "d_nom" : np.array([0, 0, 0, 0, 0, 0, 0]),
                "r_nom" : np.array([0, 0, 0.42, 0, 0.4, 0, 0]),
                "alpha_nom" : np.array([0, pip2, -pip2, -pip2, pip2, pip2, -pip2])}

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
        self.rls = RLS(4*num_links, 1)
        self.distances = np.zeros((0,))



    @staticmethod
    def get_T__i(q__i, theta__i, d__i, r__i, alpha__i, type='revolute') -> np.array:
        Rx, Rz, Trans = utils.Rx, utils.Rz, utils.Trans
        if type=='revolute':
            # T = Rz(q__i+theta__i) @ Trans(0, 0, d__i) @ Trans(r__i, 0, 0) @ Rx(alpha__i)
            T = Rx(alpha__i) @ Trans(d__i, 0, 0) @ Rz(theta__i + q__i) @ Trans(0, 0, r__i)
        elif type=='prismatic':
            # T = Rz(theta__i) @ Trans(0, 0, q__i + d__i) @ Trans(r__i, 0, 0) @ Rx(alpha__i)
            T = Rx(alpha__i) @ Trans(d__i, 0, 0) @ Rz(theta__i) @ Trans(0, 0, r__i+q__i)
        else:
            return None
        return T

    @staticmethod
    def get_T_jk(j,k,q,theta_all, d_all, r_all, alpha_all) -> np.array:
        """
        T_jk = T^j_k
        """
        q, theta_all, d_all, r_all, alpha_all = q.flatten(), theta_all.flatten(), d_all.flatten(), r_all.flatten(), alpha_all.flatten()
        T = np.eye(4)
        for i in range(j, k):
            # print(f"i={i}, j={j}, k={k}, theta={theta_all[i]+q[i]}, d={d_all[i]}, r={r_all[i]}, alpha={alpha_all[i]}\n")
            _T = ParameterEstimator.get_T__i(q[i], theta_all[i], d_all[i], r_all[i], alpha_all[i])
            T = T @ _T
        return T



    def get_parameter_jacobian(self, theta_all, d_all, a_all, alpha_all) -> np.array:
        """
        Get the parameter jacobian, that is the matrix approximating the effect of parameter (DH)
        deviations on the final pose. The number of links is inferred from the lenght of the DH 
        parameter vectors. All joints are assumed rotational.
        """
        assert theta_all.size == d_all.size == a_all.size == alpha_all.size, "All parameter vectors must have same length"
        num_links = theta_all.size

        W1 = W2 = W3 = W4 = W7 = W8 = np.zeros((3,0))

        T__n_0=self.get_T__i0(num_links, theta_all, d_all, a_all, alpha_all)
        t__n_0=T__n_0[0:3,3]
        for i in range(1,num_links+1):
            T__i_0=self.get_T__i0(i-1, theta_all, d_all, a_all, alpha_all)
            t__i_0=T__i_0[0:3,3]
            R__i_0=T__i_0[0:3,0:3]
            m__1i=np.array([[0],[0],[1]])
            m__2i=np.array([[m.cos(theta_all[i-1])],[m.sin(theta_all[i-1])],[0]])
            m__3i=np.array([[-d_all[i-1]*m.sin(theta_all[i-1])],[d_all[i-1]*m.cos(theta_all[i-1])],[0]])
            t1=np.matmul(R__i_0,m__1i)
            t2=np.matmul(R__i_0,m__2i)

            w=np.reshape(np.cross(t__i_0,t1.flatten()),(3,1))

            W1 = np.concatenate((W1,w), axis=1)

            W2 = np.concatenate((W2,t1), axis=1)

            W3 = np.concatenate((W3,t2), axis=1)

            w=np.reshape(np.cross(t__i_0,t2.flatten()),(3,1))+np.matmul(R__i_0,m__3i)
            W4 = np.concatenate((W4,w),axis=1)

            w = np.reshape(np.cross(t1.flatten(),t__n_0),(3,1))+np.reshape(W1[:,-1],(3,1))
            W7 = np.concatenate((W7,w),axis=1)

            w=np.reshape(np.cross(t2.flatten(),t__n_0),(3,1))+np.reshape(W4[:,-1],(3,1))
            W8=np.concatenate((W8,w),axis=1)
        J = np.zeros((6,4*num_links))
        J[0:3,:]=np.concatenate((W7, W2, W3, W8), axis=1)
        J[3:6,:]=np.concatenate((W2, np.zeros((3,num_links)), np.zeros((3,num_links)), W3), axis=1)
        return J


    def process_measurement(self, m):
        # calculate the poses of the camera based on the nominal 
        # parameters and forward kinematics
        try:
            num_links = (np.array(m.q1)).size
            theta_nom1 = np.array(m.q1)+self.theta_nom
            T_nom1 = self.get_T__i0(num_links,theta_nom1, self.d_nom, self.r_nom, self.alpha_nom)
            tvec_nom1 = T_nom1[0:3,3].reshape((3,1))
            rvec_nom1 = cv2.Rodrigues(T_nom1[0:3,0:3])[0]


            theta_nom2 = np.array(m.q2)+self.theta_nom
            T_nom2 = self.get_T__i0(num_links,theta_nom2, self.d_nom, self.r_nom, self.alpha_nom)
            tvec_nom2 = T_nom2[0:3,3].reshape((3,1))
            rvec_nom2 = cv2.Rodrigues(T_nom2[0:3,0:3])[0]

            # calculate the difference in the nominal poses
            dtvec_nom = tvec_nom1 - tvec_nom2
            drvec_nom = rvec_nom1 - rvec_nom2
        except Exception as e:
            rospy.logerr("process_measurement: nominal calc failed: {}".format(e))
            return

        # calculate the error between th expected and measured pose differenced
        dtvec_real, drvec_real = np.reshape(np.array(m.dtvec),(3,1)), np.reshape(np.array(m.drvec),(3,1))
        current_error=np.concatenate((dtvec_real-dtvec_nom,drvec_real-drvec_nom),axis=0)

        # calculate the corresponding difference jacobian
        jacobian1 = self.get_parameter_jacobian(theta_nom1, self.d_nom, self.r_nom, self.alpha_nom)
        jacobian2 = self.get_parameter_jacobian(theta_nom2, self.d_nom, self.r_nom, self.alpha_nom)
        jacobian = jacobian1-jacobian2

        try:
            # use RLS
            self.rls.add_obs(S=jacobian, Y=current_error)
            estimate_k = self.rls.get_estimate().flatten()
        except Exception as e:
            rospy.logerr("process_measurement: RLS failed: {}".format(e))

        # compose and publish message containing estimate
        msg = Array_f64()
        msg.data = estimate_k
        msg.time = rospy.get_time()
        self.pub_est.publish(msg)


class RLS():
    def __init__(self, num_params, q, alpha=1e3)->None:
        """
        num_params: number of parameters to be estimated
        q: forgetting factor, usually very close to 1.
        alpha: initial value on diagonal of P
        """
        assert q <= 1 and q > 0.95, "q usually needs to be from ]0.95, 1]"
        self.q = q

        self.alpha=alpha
        self.num_params = num_params
        self.P = alpha*np.eye(num_params) #initial value of matrix P
        self.phat = np.zeros((num_params,1)) #initial guess for parameters, col vector
        self.num_obs=0


    def add_obs(self, S, Y)->None:
        """
        Add an observation
        S_T: array of data vectors [[s1],[s2],[s3]...]
        Y: measured outputs vector [[y1],[y2],[y3]...]
        """
        if S.ndim==1: # 1D arrays are converted to a row in a 2D array
            S = np.reshape(S,(1,-1))
        if Y.ndim==1:
            Y = np.reshape(Y,(-1,1))

        assert np.shape(S)[1]==self.num_params, "number of parameters has to agree with measurement dim"
        assert np.shape(S)[0]==np.shape(Y)[0], "observation dimensions don't match"



        for obs in zip(S,Y): # iterate over rows, each iteration is an independent measurement
            (s_T, y)=obs
            s_T = np.reshape(s_T,(1,-1))
            s=np.transpose(s_T)
            _num=self.P@s
            _den=(self.q + s_T@self.P@s)
            self.k = _num/_den

            self.P = (self.P - self.k@s_T@self.P)*(1/self.q)

            self.phat = self.phat + self.k*(y - s_T@self.phat)
            self.num_obs = self.num_obs+1

        # if (np.any(np.abs(self.phat)>0.15)): # values are too big, reset LSQ
        #     self.P = self.alpha*np.eye(self.num_params) #initial value of matrix P
        #     self.phat = np.zeros((self.num_params,1)) #initial guess for parameters, col vector

    def get_estimate(self):
        return self.phat

    def get_num_obs(self):
        return self.num_obs


# Main function.
if __name__ == "__main__":
    rospy.init_node('dh_estimator')   # init ROS node
    rospy.loginfo('#Node dh_estimator running#')

    while not rospy.get_rostime():      # wait for ros time service
        pass
    pe = ParameterEstimator()          # create instance

    rospy.spin()
