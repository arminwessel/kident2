#!/usr/bin/env python3
"""
Utility Functions
"""
import cv2
import numpy as np
import math as m

def H_rvec_tvec(rvec,tvec):
    rotmat, _ = cv2.Rodrigues(np.array(rvec).flatten())
    lower = np.reshape(np.array([0, 0, 0, 1]), (1, 4))
    upper = np.concatenate(
        (rotmat, np.reshape(np.array(tvec), (3, 1))),
        axis=1
    )
    H = np.concatenate(
        (upper, lower),
        axis=0
    )
    return np.asarray(H)

def mat2rvectvec(transformation_matrix):
    assert (4, 4) == np.shape(transformation_matrix)
    R = transformation_matrix[0:3, 0:3]
    rvec = cv2.Rodrigues(R)[0].flatten()
    # theta = np.linalg.norm(rvec)
    # while np.abs(theta) > np.pi * 0.99999:
    #     rvec = (rvec / theta) * (theta - np.pi)   # normalize with theta, then scale up to unwrapped magnitude
    #     theta = np.linalg.norm(rvec)
    tvec = transformation_matrix[0:3, 3]
    return rvec, tvec

def roundprint(H):
    for line in H:
        linestr=''
        for elem in line:
            elem = 0.001*round(1000*elem)
            if (elem>=0):
                linestr += " {:.3f} ".format(elem)
            else:
                linestr += "{:.3f} ".format(elem)
        print(linestr+'\n')
    print('\n')
    
def Rx(x):
    return np.array([   [ 1, 0       , 0       , 0],
                        [ 0, m.cos(x),-m.sin(x), 0],
                        [ 0, m.sin(x), m.cos(x), 0],
                        [ 0, 0       , 0       , 1]])

def Ry(x):
    return np.array([   [ m.cos(x), 0, m.sin(x), 0],
                        [ 0       , 1, 0       , 0],
                        [-m.sin(x), 0, m.cos(x), 0],
                        [0        , 0, 0       , 1]])

def Rz(x):
    return np.array([   [ m.cos(x), -m.sin(x), 0, 0],
                        [ m.sin(x), m.cos(x) , 0, 0],
                        [ 0       , 0        , 1, 0],
                        [ 0       , 0        , 0, 1]])

def Trans(x,y,z):
    return np.array([   [ 1, 0, 0, x],
                        [ 0, 1, 0, y],
                        [ 0, 0, 1, z],
                        [ 0, 0, 0, 1]])