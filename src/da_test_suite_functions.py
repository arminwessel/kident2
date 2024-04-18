import pickle
import numpy as np
import scipy.linalg
import sympy
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from parameter_estimator import ParameterEstimator
import utils
from itertools import combinations
import random
import time
from scipy.optimize import least_squares
from matplotlib.backends.backend_pdf import PdfPages
from pylatex import Document, NoEscape, Package
from pathlib import Path
from robot import RobotDescription
from sklearn.covariance import MinCovDet
from pytransform3d.transform_manager import *
from acin_colors import acin_colors


def get_errors(length):
    normal_errors = np.array(  [-0.0045, 0.0065, -0.0170, 0.0096, -0.0034,
                                -0.0252, 0.0025, -0.0126, 0.0165, 0.0044,
                                -0.0209, -0.0009, -0.0054, 0.0008, -0.0074,
                                0.0130, -0.0090, 0.0063, -0.0159, -0.0018,
                                0.0072, -0.0054, 0.0022, 0.0038, 0.0123,
                                -0.0001, 0.0004, 0.0127, 0.0052, -0.0041,
                                -0.0093, -0.0012, -0.0044, 0.0014, -0.0032,
                                -0.0066, -0.0029, 0.0002, -0.0102, -0.0063,
                                -0.0001, 0.0126, 0.0001, -0.0171, 0.0152,
                                -0.0041, 0.0190, 0.0142, -0.0185, -0.0041,
                                0.0032, 0.0010, 0.0055, 0.0088, -0.0064,
                                0.0065, 0.0070, -0.0096, 0.0213, -0.0052,
                                0.0146, -0.0035, -0.0032, -0.0035, 0.0069,
                                -0.0119, 0.0030, -0.0025, 0.0232, 0.0112,
                                0.0152, -0.0116, -0.0074, -0.0132, -0.0015,
                                0.0089, -0.0064, 0.0017, 0.0108, -0.0000,
                                0.0107, 0.0013, 0.0041, -0.0039, 0.0037,
                                -0.0048, -0.0016, -0.0140, -0.0064, -0.0035,
                                0.0174, -0.0006, 0.0067, -0.0158, 0.0020,
                                0.0131, -0.0096, 0.0285, 0.0161, 0.0050])
    return np.random.choice(normal_errors, length)


def apply_error_to_params(nominal_values, mask, errors, factor):
    """
    Adds scaled errors to nominal parameters, so that robot configurations can be simulated with defined errors.
    The mask allows to enable/disable the identification of each parameter
    """
    assert len(mask) == len(nominal_values)

    param_with_errors = [value + factor * error if apply else value for apply, value, error in zip(mask,
                                                                                                   nominal_values,
                                                                                                   errors)]
    return param_with_errors


def residuals(params, errors_tot, jacobian_tot):
    """
    Calculates the residual error after LM parameter estimation
    """
    ret = errors_tot - jacobian_tot @ params
    return ret


def asSpherical(x, y, z):
    r = m.sqrt(x*x + y*y + z*z)
    theta = m.acos(z/r)*180 / m.pi  # to degrees
    phi = m.atan2(y, x)*180 / m.pi
    return r, theta, phi


def asCartesian(r, theta, phi):
    theta = theta * m.pi/180  # to radian
    phi = phi * m.pi/180
    x = r * m.sin(theta) * m.cos(phi)
    y = r * m.sin(theta) * m.sin(phi)
    z = r * m.cos(theta)
    return x, y, z


def create_pairs_random(input_list):
    pairs = []
    while len(input_list) > 1:
        element1 = input_list.pop(random.randrange(len(input_list)))
        element2 = input_list.pop(random.randrange(len(input_list)))
        pairs.append((element1, element2))
    return pairs


def identify(obs_pairs, expected_parameters, parameter_id_masks, method='lm'):
    """
    Function estimating the parameter error based on a linear model

    obs_pairs: a list of observation pairs
    k_obs: number of observations to be used, passed to get_jacobian
    expected_parameters: these parameters will be used to generate the jacobian, the error calculated in each iteration
    of identify is expressed with respect to these parameters

    """
    # parameters to use for calculating jacobian
    theta = np.array(expected_parameters['theta'])
    d = np.array(expected_parameters['d'])
    r = np.array(expected_parameters['r'])
    alpha = np.array(expected_parameters['alpha'])

    array_expected_params = np.concatenate((theta, d, r, alpha))
    num_params = len(array_expected_params)
    positions = np.arange(num_params)
    param_names = [p+'_'+str(i) for p in ['theta', 'd', 'r', 'alpha'] for i in range(len(theta))]

    # calculate jacobian
    jacobian_tot, errors_tot, jac_quality = RobotDescription.get_linear_model(obs_pairs, theta, d, r, alpha, True)

    # number of parameters to identify
    total_id_mask = (parameter_id_masks['theta'] + parameter_id_masks['d']
                     + parameter_id_masks['r'] + parameter_id_masks['alpha'])
    num_to_ident = sum(bool(x) for x in total_id_mask)  # number of parameters to identify

    # delete the columns where the mask is false - this is so that only the parameters marked with True are identified
    jacobian_tot_reduced = np.delete(jacobian_tot, np.where(np.logical_not(total_id_mask)), axis=1)
    ##########
    mat_q, mat_r = np.linalg.qr(jacobian_tot_reduced)
    jac_quality['qr_diag_r_reduced_jacobian'] = np.diagonal(mat_r)
    jac_quality['rank_full_jacobian'] = np.linalg.matrix_rank(jacobian_tot_reduced)
    jac_quality['svdvals'] = scipy.linalg.svdvals(jacobian_tot)
    jac_quality['svdvals_reduced_jacobian'] = scipy.linalg.svdvals(jacobian_tot_reduced)


    ##########
    expected_parameters_reduced = np.delete(array_expected_params, np.where(np.logical_not(total_id_mask)))
    positions_reduced = np.delete(positions, np.where(np.logical_not(total_id_mask)))
    param_names_reduced = np.delete(param_names, np.where(np.logical_not(total_id_mask)))

    if method == 'lsq':
        errors_reduced, _, _, _ = np.linalg.lstsq(jacobian_tot_reduced, errors_tot)
    elif method == 'lm':
        res = least_squares(fun=residuals,
                            x0=expected_parameters_reduced, method='lm', args=(errors_tot, jacobian_tot_reduced))
        errors_reduced = res.x
    else:
        print("No valid option for solver chosen in identify")

    array_errors = np.zeros(num_params)  # initialize
    array_errors[positions_reduced] = errors_reduced  # insert the identified errors at their original positions
    array_estimated_params = array_expected_params + array_errors  # add the error to the expected params

    error_theta, error_d, error_r, error_alpha = np.split(array_errors, 4)
    estimated_errors = {'theta': error_theta, 'd': error_d, 'r': error_r, 'alpha': error_alpha}
    est_theta, est_d, est_r, est_alpha = np.split(array_estimated_params, 4)
    estimated_params = {'theta': est_theta, 'd': est_d, 'r': est_r, 'alpha': est_alpha}

    additional_info = {'param_names_reduced': param_names_reduced,
                       'param_names': param_names,
                       'jac_quality': jac_quality,
                       'residuals': residuals(errors_reduced, errors_tot, jacobian_tot_reduced),
                       'method_used': method}
    return estimated_errors, estimated_params, additional_info


def acin_color_palette():
    # import colors for plotting
    f_acin_colors = open('acin_colors.p', 'rb')
    acin_colors = pickle.load(f_acin_colors)
    f_acin_colors.close()

    colors = np.array([acin_colors['acin_red'], acin_colors['acin_green'], acin_colors['acin_yellow'],
                       acin_colors['TU_blue'], acin_colors['acin_yellow_variant'],
                       acin_colors['acin_green_variant'], acin_colors['acin_blue_variant'],
                       acin_colors['TU_pink_variant']])
    index = -1
    while True:
        index += 1
        yield colors[index % len(colors)]


def diff_dictionaries(dict1, dict2):
    dict_res = {}
    for key in dict1:
        dict_res[key] = np.array(dict1[key]) - np.array(dict2[key])
    return dict_res


def result_to_df(dict_error, dict_result):
    df = pd.DataFrame()
    errors, errors_mm_deg, results, results_mm_deg, tex_names = [], [], [], [], []
    num_params = len(dict_error['theta'])
    for i in range(num_params):
        errors.append(dict_error['theta'][i])
        errors.append(dict_error['d'][i])
        errors.append(dict_error['r'][i])
        errors.append(dict_error['alpha'][i])

        errors_mm_deg.append(dict_error['theta'][i] / np.pi * 180)
        errors_mm_deg.append(dict_error['d'][i] * 1000)
        errors_mm_deg.append(dict_error['r'][i] * 1000)
        errors_mm_deg.append(dict_error['alpha'][i] / np.pi * 180)

        results.append(dict_result['theta'][i])
        results.append(dict_result['d'][i])
        results.append(dict_result['r'][i])
        results.append(dict_result['alpha'][i])

        results_mm_deg.append(dict_result['theta'][i] / np.pi * 180)
        results_mm_deg.append(dict_result['d'][i] * 1000)
        results_mm_deg.append(dict_result['r'][i] * 1000)
        results_mm_deg.append(dict_result['alpha'][i] / np.pi * 180)

        tex_names.append('$\\theta_{' + str(i) + '}$ [$\\degree$]')
        tex_names.append('$d_{' + str(i) + '}$ [mm]')
        tex_names.append('$r_{' + str(i) + '}$ [mm]')
        tex_names.append('$\\alpha_{' + str(i) + '}$ [$\\degree$]')

    errors = np.array(errors)
    results = np.array(results)
    errors_mm_deg = np.array(errors_mm_deg)
    results_mm_deg = np.array(results_mm_deg)
    df['tex_names'] = tex_names
    df['errors'] = errors
    df['results'] = results
    df['results_mm_deg'] = results_mm_deg
    df['errors_mm_deg'] = errors_mm_deg
    df['identification_accuracy'] = df['results_mm_deg'] - df['errors_mm_deg']
    return df


def plot_evolution(titles, evolution_list):

    fig, ax_est = plt.subplots(2, 2)
    fig.set_size_inches(8, 4.5, forward=True)
    fig.tight_layout(pad=2)
    fig.subplots_adjust(top=0.85)
    fig.suptitle(titles['suptitle'])

    color_palette = acin_color_palette()

    # sort list of dicts into dict of lists
    evolution_dict = {param: np.array([dic[param] for dic in evolution_list]) for param in evolution_list[0]}

    axis = ax_est

    axis[0, 0].clear()
    for i, Y in enumerate(evolution_dict['theta'].T):  # first Y is the evolution of theta 0 over time
        Y = Y * 180 / np.pi
        X = range(len(Y))
        axis[0, 0].plot(X, Y, color=next(color_palette), label=str(i))
    axis[0, 0].set_title(titles['theta'])
    # axis[0, 0].legend()

    axis[0, 1].clear()
    for i, Y in enumerate(evolution_dict['d'].T):  # first Y is the evolution of theta 0 over time
        Y = Y * 1000
        X = range(len(Y))
        axis[0, 1].plot(X, Y, color=next(color_palette), label=str(i))
    axis[0, 1].set_title(titles['d'])
    # axis[0, 1].legend()

    axis[1, 0].clear()
    for i, Y in enumerate(evolution_dict['r'].T):  # first Y is the evolution of theta 0 over time
        Y = Y * 1000
        X = range(len(Y))
        axis[1, 0].plot(X, Y, color=next(color_palette), label=str(i))
    axis[1, 0].set_title(titles['r'])
    axis[1, 0].set_xlabel("iterations")
    # axis[1, 0].legend()


    axis[1, 1].clear()
    for i, Y in enumerate(evolution_dict['alpha'].T):  # first Y is the evolution of theta 0 over time
        Y = Y * 180 / np.pi
        X = range(len(Y))
        axis[1, 1].plot(X, Y, color=next(color_palette), label=str(i))
    axis[1, 1].set_title(titles['alpha'])
    axis[1, 1].set_xlabel("iterations")
    # axis[1, 1].legend()

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.23)  ##  Need to play with this number.

    labels = [f"link {i}" for i in range(1, len(evolution_dict['theta'].T)+1)]
    fig.legend(labels=labels, loc="lower center", ncol=4)

    return fig


def _draw_as_table(df, pagesize):
    alternating_colors = [['white'] * len(df.columns), ['lightgray'] * len(df.columns)] * len(df)
    alternating_colors = alternating_colors[:len(df)]
    fig, ax = plt.subplots(figsize=pagesize)
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values,
                         rowLabels=df.index,
                         colLabels=df.columns,
                         rowColours=['lightblue'] * len(df),
                         colColours=['lightblue'] * len(df.columns),
                         cellColours=alternating_colors,
                         loc='center')
    return fig

def latex_to_pdf(filepath, filename, content):
    default_filepath = Path(filepath).joinpath(filename)
    doc = Document(default_filepath=default_filepath,
                   documentclass='standalone')
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('gensymb'))
    doc.append(NoEscape(content))
    doc.generate_pdf(clean_tex=False)

def dataframe_to_pdf(df, filename, numpages=(1, 1), pagesize=(11, 8.5)):
    with PdfPages(filename) as pdf:
        nh, nv = numpages
        rows_per_page = len(df) // nh
        cols_per_page = len(df.columns) // nv
        for i in range(0, nh):
            for j in range(0, nv):
                page = df.iloc[(i * rows_per_page):min((i + 1) * rows_per_page, len(df)),
                       (j * cols_per_page):min((j + 1) * cols_per_page, len(df.columns))]
                fig = _draw_as_table(page, pagesize)
                if nh > 1 or nv > 1:
                    # Add a part/page number at bottom-center of page
                    fig.text(0.5, 0.5 / pagesize[0],
                             "Part-{}x{}: Page-{}".format(i + 1, j + 1, i * nv + j + 1),
                             ha='center', fontsize=8)
                pdf.savefig(fig, bbox_inches='tight')

                plt.close()


def get_marker_locations(df):
    marker_locations = []
    theta_nom = RobotDescription.dhparams["theta_nom"].astype(float)
    d_nom = RobotDescription.dhparams["d_nom"].astype(float)
    r_nom = RobotDescription.dhparams["r_nom"].astype(float)
    alpha_nom = RobotDescription.dhparams["alpha_nom"].astype(float)
    for record in df.to_records():
        q = np.concatenate([record['q'], np.zeros(RobotDescription.dhparams['num_cam_extrinsic'])])
        _loc = RobotDescription.get_marker_location(record['mat'], q, theta_nom, d_nom, r_nom, alpha_nom)
        marker_location = RobotDescription.get_alternate_tfs(_loc)
        marker_locations.append(np.concatenate([marker_location['rvec'], marker_location['tvec']]))
    return marker_locations


def split_df_by_marker_id(df):
    return [df[df['marker_id']==marker_id].copy() for marker_id in df['marker_id'].unique()]


def reject_outliers_by_mahalanobis_dist(dataframe, threshold):
    df_filtered = pd.DataFrame()
    single_marker_dfs = split_df_by_marker_id(dataframe)
    for df in single_marker_dfs:
        locations = get_marker_locations(df)
        cov_matrix = np.cov(locations)
        rank_cov_matrix = np.linalg.matrix_rank(cov_matrix)
        if rank_cov_matrix + 1 < np.shape(locations)[1]:
            # not full rank
            df_filtered = pd.concat([df_filtered, df])
        else:
            robust_cov = MinCovDet().fit(locations)
            df.loc[:, 'mahal_dist'] = robust_cov.dist_
            df = df[df['mahal_dist'] < threshold]
            df_filtered = pd.concat([df_filtered, df])
    df_filtered.reset_index()
    return df_filtered


def get_joint_tfs_pose_err_plot(q_vec, params):
    """
    modified version for the one plot function only
    """
    joint_tfs = []  # initialize list
    q_vec = q_vec.flatten()
    q_vec = np.append(q_vec, np.zeros(RobotDescription.dhparams["num_cam_extrinsic"]))  # pad q vector with zero for non actuated last transform
    for (i, q) in enumerate(q_vec):  # iterate over joint values
        theta = params["theta"][i]
        d = params["d"][i]
        r = params["r"][i]
        alpha = params["alpha"][i]
        joint_tfs.append({'mat': RobotDescription.get_T_i_forward(q, theta, d, r, alpha),
                          'from_frame': str(i+1), 'to_frame': str(i)})

    joint_tfs.append({'mat': RobotDescription.T_W0, 'from_frame': '0', 'to_frame': 'world'})
    return joint_tfs


def plot_pose_errors_dist(nominal_parameters, error_parameters, estd_parameters, df_observations):
    # len of robot
    last_frame = RobotDescription.dhparams['num_cam_extrinsic'] + RobotDescription.dhparams['num_joints']

    # extract all qs from observations
    qs = np.vstack(df_observations['q'].to_numpy())
    qs_unique = np.unique(qs, axis=0)

    # create dataframe to store data
    data = pd.DataFrame()
    data['q'] = list(qs_unique)

    list_nom_positions = []
    list_err_positions = []
    list_est_positions = []

    for q in data['q']:
        # get camera poses nominal
        joint_tfs = get_joint_tfs_pose_err_plot(q, nominal_parameters)
        tm = TransformManager()
        for tf in joint_tfs:
            from_frame, to_frame, A2B = tf['from_frame'], tf['to_frame'], tf['mat']
            tm.add_transform(from_frame, to_frame, A2B)
        list_nom_positions.append(np.array(tm.get_transform(str(last_frame), 'world'))[0:3, 3])

        # get camera poses error
        joint_tfs = get_joint_tfs_pose_err_plot(q, error_parameters)
        tm = TransformManager()
        for tf in joint_tfs:
            from_frame, to_frame, A2B = tf['from_frame'], tf['to_frame'], tf['mat']
            tm.add_transform(from_frame, to_frame, A2B)
        list_err_positions.append(np.array(tm.get_transform(str(last_frame), 'world'))[0:3, 3])

        # get camera poses identified
        joint_tfs = get_joint_tfs_pose_err_plot(q, estd_parameters)
        tm = TransformManager()
        for tf in joint_tfs:
            from_frame, to_frame, A2B = tf['from_frame'], tf['to_frame'], tf['mat']
            tm.add_transform(from_frame, to_frame, A2B)
        list_est_positions.append(np.array(tm.get_transform(str(last_frame), 'world'))[0:3, 3])

    data['nominal'] = list_nom_positions
    data['error'] = list_err_positions
    data['estd'] = list_est_positions

    # calculate the distances
    data['dist_err'] = data['nominal'] - data['error']
    data['dist_err'] = data['dist_err'].apply(np.linalg.norm)
    # print(data['dist_err'][0])

    data['dist_est'] = data['nominal'] - data['estd']
    data['dist_est'] = data['dist_est'].apply(np.linalg.norm)
    # print(data['dist_est'][0])

    data['index'] = data.index

    data['dist_err_mm'], data['dist_est_mm'] = data['dist_err'] * 1000, data['dist_est'] * 1000

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    data.plot.scatter(ax=ax, x='index', y='dist_err_mm', color=acin_colors['red'])
    data.plot.scatter(ax=ax, x='index', y='dist_est_mm', color=acin_colors['blue'])
    ax.set_xlabel('Number of configuration')
    ax.set_ylabel('Distance Error [mm]')

    ax.legend(['uncalibrated', 'calibrated'], bbox_to_anchor=(0.5, 1.0), loc='lower center')

    return fig


def autolabel_plot_pose_errors_bar(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_pose_errors_bar(labels, xlabel, ylabel, err_max, err_mean):
    err_max, err_mean = np.array(err_max) * 1000, np.array(err_mean) * 1000
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(4, 3))
    rects1 = ax.bar(x - width / 2, err_max, width, label='max', color=acin_colors['blue'])
    rects2 = ax.bar(x + width / 2, err_mean, width, label='mean', color=acin_colors['green'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)
    ax.legend()

    # autolabel_plot_pose_errors_bar(rects1, ax)
    # autolabel_plot_pose_errors_bar(rects2, ax)

    fig.tight_layout()

    return fig
