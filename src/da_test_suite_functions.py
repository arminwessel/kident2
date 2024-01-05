import pickle
import numpy as np
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

def apply_error_to_params(nominal_values, mask, factor, convert):
    """
    Adds scaled errors to nominal parameters, so that robot configurations can be simulated with defined errors.
    The mask allows to enable/disable the identification of each parameter
    """
    # normal distribution around 0 with sigma=1
    normal_dist = np.array([0.1813318,  0.88292639,  0.67721319, -0.94740093,
                            -0.45521845, -0.38626203,  0.85088985, -0.49172579])
    if convert == 'm_to_mm':
        normal_dist_converted = normal_dist/1000
    elif convert == 'deg_to_rad':
        normal_dist_converted = normal_dist/180*np.pi
    else:
        print('no conversion given')
        normal_dist_converted = normal_dist
    param_with_errors = [value + factor * error if apply else value for apply, value, error in zip(mask, nominal_values, normal_dist_converted)]
    return param_with_errors


def residuals(params, errors_tot, jacobian_tot):
    """
    Calculates the residual error after LM parameter estimation
    """
    ret = errors_tot - jacobian_tot @ params
    return ret


# def get_obs_distance(obs1, obs2):
    

def get_jacobian(observations, theta, d, r, alpha, k_obs):
    """
    Iterate over all observations. For each pair, compute the relative transform between them,
    and from that calculate the error vector and the corresponding jacobian.
    Collect all jacobians and error vectors to create the identification dataset
    Additionally, the marker locations are returned to check if marker localization works correctly
    """
    pe = ParameterEstimator()

    num_params = len(theta) + len(d) + len(r) + len(alpha)
    jacobian_tot = np.zeros((0, num_params))
    errors_tot = np.zeros(0)

    # initilalize a list to contain the observed location of the markers in world coordinates
    list_marker_locations = []

    # loop over all markers in observations
    for markerid in list(observations)[:]:
        num_observed = 0
        count = 0

        # compute all possible pairs from observations of this marker
        comparisons = []
        for obs1, obs2 in combinations(observations[markerid], 2):
            count = count+1
            comparisons.append((obs1, obs2))


        # randomly choose a limited number of these comparisons except if k_obs == 'all'
        if k_obs == 'all':
            comparisons_reduced = comparisons
        else:
            comparisons_reduced = random.choices(comparisons, k=k_obs)

        # iterate over the observation pairs
        for obs1, obs2 in comparisons_reduced:

            # extract measurements
            q1 = np.hstack((np.array(obs1["q"]), np.zeros(1)))
            q2 = np.hstack((np.array(obs2["q"]), np.zeros(1)))
            # transform the coordinate systems from ROS converntion to OpenCV convention
            T_CM_1 = pe.T_corr @ utils.H_rvec_tvec(obs1["rvec"], obs1["tvec"]) @ np.linalg.inv(pe.T_corr) #@ pe.T_correct_cam_mdh
            T_CM_2 = pe.T_corr @ utils.H_rvec_tvec(obs2["rvec"], obs2["tvec"]) @ np.linalg.inv(pe.T_corr) #@ pe.T_correct_cam_mdh

            # calculate nominal transforms
            T_08_1 = pe.get_T_jk(0, 8, q1, theta, d, r, alpha)
            T_08_2 = pe.get_T_jk(0, 8, q2, theta, d, r, alpha)

            # perform necessary inversions
            T_MC_2 = np.linalg.inv(T_CM_2)
            T_80_1 = np.linalg.inv(T_08_1)

            T_WM_1 = pe.T_W0 @ T_08_1 @ T_CM_1
            if markerid == 2 or True:
                list_marker_locations.append([T_WM_1[0, 3], T_WM_1[1, 3], T_WM_1[2, 3]])

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

            # collect the jacobian and error resulting from these two observations
            jacobian_tot = np.concatenate((jacobian_tot, jacobian), axis=0)
            errors_tot = np.concatenate((errors_tot, pose_error), axis=0)

    mat_q, mat_r = np.linalg.qr(jacobian_tot)
    diag_r = np.diagonal(mat_r)
    rank = np.linalg.matrix_rank(jacobian_tot)
    jacobian_quality = {'qr_criterion': diag_r, 'rank': rank}

    return jacobian_tot, errors_tot, list_marker_locations, jacobian_quality


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


def identify(observations, k_obs, expected_parameters, parameter_id_masks, method='lm'):
    """
    Function estimating the parameter error based on a linear model

    observations: a dictionary of observations of the real robot to use in estimating the real parameters
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

    # calculate jacobian
    jacobian_tot, errors_tot, list_marker_locations, jac_quality = get_jacobian(observations, theta, d, r, alpha, k_obs)


    # number of parameters to identify
    total_id_mask = (parameter_id_masks['theta'] + parameter_id_masks['d']
                     + parameter_id_masks['r'] + parameter_id_masks['alpha'])
    num_to_ident = sum(bool(x) for x in total_id_mask)  # number of parameters to identify

    # delete the columns where the mask is false - this is so that only the parameters marked with True are identified
    jacobian_tot_reduced = np.delete(jacobian_tot, np.where(np.logical_not(total_id_mask)), axis=1)
    ##########
    mat_q, mat_r = np.linalg.qr(jacobian_tot_reduced)
    diag_r = np.diagonal(mat_r)
    rank = np.linalg.matrix_rank(jacobian_tot_reduced)
    ##########
    expected_parameters_reduced = np.delete(array_expected_params, np.where(np.logical_not(total_id_mask)))
    positions_reduced = np.delete(positions, np.where(np.logical_not(total_id_mask)))

    if method == 'lsq':
        errors_reduced, _, _, _ = np.linalg.lstsq(jacobian_tot_reduced, errors_tot)
    else:
        res = least_squares(fun=residuals,
                            x0=expected_parameters_reduced, method='lm', args=(errors_tot, jacobian_tot_reduced))
        errors_reduced = res.x

    array_errors = np.zeros(num_params)  # initialize
    array_errors[positions_reduced] = errors_reduced  # insert the identified errors at their original positions
    array_estimated_params = array_expected_params + array_errors  # add the error to the expected params

    error_theta, error_d, error_r, error_alpha = np.split(array_errors, 4)
    estimated_errors = {'theta': error_theta, 'd': error_d, 'r': error_r, 'alpha': error_alpha}
    est_theta, est_d, est_r, est_alpha = np.split(array_estimated_params, 4)
    estimated_params = {'theta': est_theta, 'd': est_d, 'r': est_r, 'alpha': est_alpha}

    additional_info = {'marker_locations': list_marker_locations,
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
    fig.set_size_inches(16, 9, forward=True)
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
    axis[0, 0].legend()

    axis[0, 1].clear()
    for i, Y in enumerate(evolution_dict['d'].T):  # first Y is the evolution of theta 0 over time
        Y = Y * 1000
        X = range(len(Y))
        axis[0, 1].plot(X, Y, color=next(color_palette), label=str(i))
    axis[0, 1].set_title(titles['d'])
    axis[0, 1].legend()

    axis[1, 0].clear()
    for i, Y in enumerate(evolution_dict['r'].T):  # first Y is the evolution of theta 0 over time
        Y = Y * 1000
        X = range(len(Y))
        axis[1, 0].plot(X, Y, color=next(color_palette), label=str(i))
    axis[1, 0].set_title(titles['r'])
    axis[1, 0].legend()


    axis[1, 1].clear()
    for i, Y in enumerate(evolution_dict['alpha'].T):  # first Y is the evolution of theta 0 over time
        Y = Y * 180 / np.pi
        X = range(len(Y))
        axis[1, 1].plot(X, Y, color=next(color_palette), label=str(i))
    axis[1, 1].set_title(titles['alpha'])
    axis[1, 1].legend()

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
