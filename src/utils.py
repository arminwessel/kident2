#!/usr/bin/env python3
"""
Utility Functions
"""
import cv2
import numpy as np
import math as m
import pickle
import pandas as pd


def save_df_to_pickle(df, filename):
    df_as_records = df.to_dict('records')
    observations_file = open(filename, 'wb')
    pickle.dump(df_as_records, observations_file)
    observations_file.close()
    print("file saved as {}".format(filename))


def open_df_from_pickle(filename):
    observations_file = open(filename, 'rb')
    df_as_records = pickle.load(observations_file)
    observations_file.close()
    return pd.DataFrame(df_as_records)



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
    tvec = transformation_matrix[0:3, 3]
    return rvec, tvec

def split_H_transform(H):
    assert (4, 4) == np.shape(H)
    R = H[0:3, 0:3]
    tvec = H[0:3, 3]
    return R, tvec

def merge_H_transform(R, tvec):
    assert (3, 3) == np.shape(R)
    tvec = np.array(tvec).flatten()
    lower = np.reshape(np.array([0, 0, 0, 1]), (1, 4))
    upper = np.concatenate(
        (R, np.reshape(np.array(tvec), (3, 1))),
        axis=1
    )
    H = np.concatenate(
        (upper, lower),
        axis=0
    )
    return H


def get_interp_distance(array, value):
    """
    find the distance between value the nearest element of the array
    """
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return np.abs(array[idx]-value)


def interpolate_vector(t0, meas_times, meas_values):
    """
    t0 time at which to evaluate each data series
    meas_times time series of measurements
    meas_values (n,m) array of n measurement series with m values each
    """
    f0 = np.zeros(np.shape(meas_values)[0])
    for i, data_series in enumerate(meas_values):
        f0[i] = np.interp(t0, meas_times, data_series)
    return f0


def roundprint(H, mode='print'):
    string = ''
    for line in H:
        linestr=''
        for elem in line:
            elem = 0.001*round(1000*elem)
            if (elem>=0):
                linestr += " {:.3f} ".format(elem)
            else:
                linestr += "{:.3f} ".format(elem)
        string = string+ linestr + '\n'
    string = string + '\n'
    if mode=='string':
        return string
    else:
        print(string)
    
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

def toHomogeneous(rotmat, trans):
    lower = np.reshape(np.array([0, 0, 0, 1]), (1, 4))
    upper = np.concatenate(
        (rotmat, np.reshape(np.array(trans), (3, 1))),
        axis=1)
    H = np.concatenate(
        (upper, lower),
        axis=0)
    return H