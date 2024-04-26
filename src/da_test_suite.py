import datetime

import numpy as np

from da_test_suite_functions import *
from exp_data_handler import *
from robot import RobotDescription


def do_experiment(parameter_id_masks, errors, factor, observations_file_select, observations_file_str_dict,
                  num_iterations, residual_norm_tolerance,  mal_threshold):

    ##########################################################################
    # 1) Define robot parameters to be used
    ##########################################################################
    # import nominal parameters
    theta_nom = RobotDescription.dhparams["theta_nom"].astype(float)
    d_nom = RobotDescription.dhparams["d_nom"].astype(float)
    r_nom = RobotDescription.dhparams["r_nom"].astype(float)
    alpha_nom = RobotDescription.dhparams["alpha_nom"].astype(float)
    nominal_parameters = {'theta': theta_nom, 'd': d_nom, 'r': r_nom, 'alpha': alpha_nom}

    # apply the errors - model the real robot
    theta_error = apply_error_to_params(theta_nom, parameter_id_masks['theta'], errors, factor)
    d_error = apply_error_to_params(d_nom, parameter_id_masks['d'], errors, factor)
    r_error = apply_error_to_params(r_nom, parameter_id_masks['r'], errors, factor)
    alpha_error = apply_error_to_params(alpha_nom, parameter_id_masks['alpha'], errors, factor)
    error_parameters = {'theta': theta_error, 'd': d_error, 'r': r_error, 'alpha': alpha_error}

    control_error_parameters = diff_dictionaries(error_parameters, nominal_parameters)
    ##########################################################################
    # 2) Import and filter observations to be used, generate pairs
    ##########################################################################
    # import selected observations
    df_observations = pd.read_pickle(observations_file_str_dict[observations_file_select])
    num_obs_unfiltered = df_observations.shape[0]  # number of records before filtering

    # filter
    df_obs_filt = reject_outliers_by_mahalanobis_dist(df_observations, mal_threshold)
    num_obs_filtered = df_obs_filt.shape[0]  # number of records before filtering
    print(f"{num_obs_filtered} /  {num_obs_unfiltered} observations after filtering")
    filter_comment = f"filtered with mal threshold {mal_threshold}"

    # pair up the observations
    marker_ids = set(df_obs_filt['marker_id'])
    obs_pairs = []
    for marker_id in marker_ids:
        df_single_marker = df_obs_filt[df_obs_filt['marker_id'] == marker_id]
        obs_pairs.extend(create_pairs_random(df_single_marker.to_dict('records')))

    ##########################################################################
    # 3) run identification as a loop
    ##########################################################################
    current_estimate = error_parameters
    current_error_evolution = []
    estimated_params_evolution = [error_parameters]
    estimated_errors_evolution = [control_error_parameters]
    rms_residuals_evolution = []
    print('begin iterative solving process')
    rms_residuals = 1e6  # init with large number
    itervar = 0
    while rms_residuals > residual_norm_tolerance and itervar <= num_iterations:
        print(f'   pass {itervar}')
        # below the observation are describing a robot with nominal parameters, and the identification is given
        # the error parameters - this is to be able to vary the error easily
        current_errors, current_estimate, additional_info = identify(obs_pairs,
                                                                     current_estimate,
                                                                     parameter_id_masks)
        jac_quality = additional_info['jac_quality']
        residuals_i = additional_info['residuals']
        method_used = additional_info['method_used']
        param_names = additional_info['param_names']
        reduced_param_names = additional_info['param_names_reduced']
        rms_residuals = np.mean(np.power(residuals_i, 2))**0.5
        max_residuals = np.max(np.abs(residuals_i))
        rms_residuals_evolution.append(rms_residuals)
        itervar = itervar + 1
        current_error_evolution.append(current_errors)
        estimated_params_evolution.append(current_estimate)
        estimated_errors = diff_dictionaries(current_estimate, nominal_parameters)
        estimated_errors_evolution.append(estimated_errors)
        remaining_error = np.array([estimated_errors[key] for key in estimated_errors]).flatten()
        if rms_residuals < residual_norm_tolerance:
            print(f'\trms residuals {rms_residuals} < tolerance {residual_norm_tolerance}')
            break
        else:
            print(f'\trms residuals {rms_residuals} > tolerance {residual_norm_tolerance}')

    print('done')

    ############################################
    titles = {'suptitle': '',  # 'Evolution of current estimate',
              'theta': r'$\theta$ [°]',
              'd': r'd [mm]',
              'r': r'r [mm]',
              'alpha': r'$\alpha$ [°]'}
    fig_param_evolution = plot_evolution(titles, estimated_params_evolution)
    ############################################
    titles = {'suptitle': '',  # 'Evolution of residual error',
              'theta': r'$\Delta$$\theta$ [°]',
              'd': r'$\Delta$d [mm]',
              'r': r'$\Delta$r [mm]',
              'alpha': r'$\Delta$$\alpha$ [°]'}
    fig_error_evolution = plot_evolution(titles, estimated_errors_evolution)
    ############################################
    fig_p_errdists = plot_pose_errors_dist(nominal_parameters,
                                           error_parameters,
                                           current_estimate,
                                           df_observations)
    ############################################

    simulated_errors = diff_dictionaries(nominal_parameters, error_parameters)
    identified_errors = diff_dictionaries(current_estimate, error_parameters)
    df_result = result_to_df(simulated_errors, identified_errors)

    exp_handler = ExperimentDataHandler()
    exp_handler.add_figure(fig_error_evolution, 'error_evolution')
    exp_handler.add_figure(fig_param_evolution, 'param_evolution')
    exp_handler.add_figure(fig_p_errdists, f'pose_error_dists_{factor}')
    exp_handler.add_note(f"nominal params: \n{RobotDescription.dhparams}\n\n" +
                         f"filtering: {num_obs_filtered}/{num_obs_unfiltered} used\n\n" +
                         f"filter comment: {filter_comment}\n\n "
                         f"parameter identification masks\n{parameter_id_masks}\n\n" +
                         f"error factor\n{factor}\n\n" +
                         f"error set\n{errors}\n\n" +
                         f"parameters with errors: \n{error_parameters}\n\n" +
                         f"observations file: \n {observations_file_str_dict[observations_file_select]}\n\n" +
                         f"number of iterations: \n {itervar}/{num_iterations} \n\n " +
                         f"residual_norm_tolerance: \n {residual_norm_tolerance}\n\n" +
                         f"param names: \n {param_names}\n\n" +
                         f"identified param names: \n {reduced_param_names}\n\n" +
                         f"method used \n {method_used}\n\n" +
                         f"jacobian quality\n{jac_quality}\n\n",
                         'info')
    exp_handler.add_note(f"norm residuals evolution \n{rms_residuals_evolution}\n\n", 'norm_convergence')
    exp_handler.add_note(f"{additional_info['jac_quality']['qr_diag_r_reduced_jacobian']}", "qr_diag")
    exp_handler.add_note(f"{np.sqrt(df_result['identification_accuracy'].pow(2).mean())}", 'residual_error_RMS')
    exp_handler.add_df(df_result, 'data')
    # exp_handler.add_marker_location(list_marker_locations[-1], 'last_marker_locations')
    exp_handler.save_experiment(r'/home/armin/catkin_ws/src/kident2/src/exp')

    ret = {'max_residuals': max_residuals, 'rms_residuals': rms_residuals}

    return ret

########################### SETTINGS ###########################
# define which parameters are to be identified
parameter_id_masks = dict()
parameter_id_masks['theta'] =   [False, True, True, True, True, True, True, True]
parameter_id_masks['d'] =       [False, True, True, True, True, True, True, True]
parameter_id_masks['r'] =       [False, True, True, True, True, True, True, True]
parameter_id_masks['alpha'] =   [False, True, True, True, True, True, True, True]


# set scaling factor for error
#factor = 0 # set in loop

# select observations file
#observations_file_select = 21 # set in loop

observations_file_str_dict = {0:  r'observation_files/ground_truth_dataset.p',
                              1:  r'observation_files/obs_exp_26_04_001_2024-04-26-11-19-23_20240426-112528.p',
                              3:  r'observation_files/obs_exp_26_04_002_2024-04-26-12-06-53_20240426-123809.p',
                              20: r'observation_files/obs_single_marker_2023-11-01-11-12-21_20240109-060457.p',
                              21: r'observation_files/observations_simulated_20240411_144805.p'
                              }


# set maximal number of iterations
num_iterations = 20

# set filter threshold
mal_threshold = 20

# tolerance to break loop
residual_norm_tolerance = 1e-6
#################################################################
# for observations_file_select in [1, 4, 9, 10]:
#     for factor in [0, 10, 20, 30, 40, 50, 60, 70]:
#         do_experiment(parameter_id_masks, factor, observations_file_select,
#                       observations_file_str_dict, num_iterations, residual_norm_tolerance)

bar_plot_data_max_mean = {'factor': [], 'mean': [], 'max': []}
errors = get_errors(len(parameter_id_masks['alpha']))

for observations_file_select in [3]:
    for factor in [1]:
        ret = do_experiment(parameter_id_masks, errors, factor, observations_file_select,
                            observations_file_str_dict, num_iterations, residual_norm_tolerance, mal_threshold)
#         bar_plot_data_max_mean['factor'].append(factor)
#         bar_plot_data_max_mean['mean'].append(ret['rms_residuals'])
#         bar_plot_data_max_mean['max'].append(ret['max_residuals'])
#
# fig_bar = plot_pose_errors_bar(bar_plot_data_max_mean['factor'],
#                                "Added Noise Factor",
#                                "Residuals",
#                                bar_plot_data_max_mean['max'],
#                                bar_plot_data_max_mean['mean'])
# exp_handler = ExperimentDataHandler()
# exp_handler.add_figure(fig_bar, 'bar_plots')
# exp_handler.save_experiment(r'/home/armin/catkin_ws/src/kident2/src/exp')