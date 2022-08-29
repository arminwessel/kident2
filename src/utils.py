#!/usr/bin/env python3
"""
Utility Functions
"""
import numpy as np
import math as m

def H(rot_euler,trans):
    rotmat_H=Rx(rot_euler[0]/180*np.pi)@Ry(rot_euler[1]/180*np.pi)@Rz(rot_euler[2]/180*np.pi)
    rotmat=rotmat_H[0:3,0:3]
    lower=np.reshape(np.array([0,0,0,1]),(1,4))
    upper=np.concatenate(
        (rotmat, np.reshape(np.array(trans),(3,1))),
        axis=1
    )
    H=np.concatenate(
        (upper, lower),
        axis=0
    )
    return np.asarray(H)


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