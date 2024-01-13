import datetime
from da_test_suite_functions import *
from exp_data_handler import *
from robot import RobotDescription


def do_experiment(parameter_id_masks, factor, observations_file_select, observations_file_str_dict,
                  num_iterations, residual_norm_tolerance):

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
    theta_error = apply_error_to_params(theta_nom, parameter_id_masks['theta'], factor, 'deg_to_rad')
    d_error = apply_error_to_params(d_nom, parameter_id_masks['d'], factor, 'm_to_mm')
    r_error = apply_error_to_params(r_nom, parameter_id_masks['r'], factor, 'm_to_mm')
    alpha_error = apply_error_to_params(alpha_nom, parameter_id_masks['alpha'], factor, 'deg_to_rad')
    error_parameters = {'theta': theta_error, 'd': d_error, 'r': r_error, 'alpha': alpha_error}

    ##########################################################################
    # 2) Import and filter observations to be used, generate pairs
    ##########################################################################
    # import selected observations
    df_observations = pd.read_pickle(observations_file_str_dict[observations_file_select])
    num_obs_unfiltered = df_observations.shape[0]  # number of records before filtering

    # filter
    df_obs_filt = reject_outliers_by_mahalanobis_dist(num_obs_unfiltered)
    num_obs_filtered = df_obs_filt.shape[0]  # number of records before filtering
    filter_comment = ""

    # pair up the observations
    marker_ids = set(df_obs_filt['marker_id'])
    obs_pairs = []
    for marker_id in marker_ids:
        df_single_marker = df_obs_filt[df_obs_filt['marker_id'] == marker_id]
        obs_pairs.extend(create_pairs_random(df_single_marker.to_dict('records')))

    ##########################################################################
    # 3) Define which parameters should be identified
    ##########################################################################
    # number of parameters to identify
    total_id_mask = (parameter_id_masks['theta'] + parameter_id_masks['d']
                     + parameter_id_masks['r'] + parameter_id_masks['alpha'])
    num_to_ident = sum(bool(x) for x in total_id_mask)  # number of parameters to identify

    # run identification as a loop
    current_estimate = error_parameters
    current_error_evolution = []
    estimated_params_evolution = []
    estimated_errors_evolution = []
    norm_residuals_evolution = []
    print('begin iterative solving process')
    # todo: maybe do a while loop until threshold, and an extra statement for a max number of iterations
    for itervar in range(num_iterations):
        print(f'   pass {itervar}')
        # below the observation are describing a robot with nominal parameters, and the identification is given
        # the error parameters - this is to be able to vary the error easily
        current_errors, current_estimate, additional_info = identify(obs_pairs,
                                                                     current_estimate,
                                                                     parameter_id_masks)
        jac_quality = additional_info['jac_quality']
        residuals_i = additional_info['residuals']
        method_used = additional_info['method_used']
        norm_residuals = np.linalg.norm(residuals_i)
        norm_residuals_evolution.append(norm_residuals)
        current_error_evolution.append(current_errors)
        estimated_params_evolution.append(current_estimate)
        estimated_errors = diff_dictionaries(current_estimate, nominal_parameters)
        estimated_errors_evolution.append(estimated_errors)
        remaining_error = np.array([estimated_errors[key] for key in estimated_errors]).flatten()
        if norm_residuals < residual_norm_tolerance:
            print(f'\tresiduals {norm_residuals} < tolerance {residual_norm_tolerance}')
            break
        else:
            print(f'\tresiduals {norm_residuals} > tolerance {residual_norm_tolerance}')

    print('done')

    ############################################
    titles = {'suptitle': 'Evolution of current estimate',
              'theta': r'$\theta$ [°]',
              'd': r'd [mm]',
              'r': r'r [mm]',
              'alpha': r'$\alpha$ [°]'}
    fig_param_evolution = plot_evolution(titles, estimated_params_evolution)
    ############################################
    titles = {'suptitle': 'Evolution of residual error',
              'theta': r'$\Delta$$\theta$ [°]',
              'd': r'$\Delta$d [mm]',
              'r': r'$\Delta$r [mm]',
              'alpha': r'$\Delta$$\alpha$ [°]'}
    fig_error_evolution = plot_evolution(titles, estimated_errors_evolution)
    ############################################

    simulated_errors = diff_dictionaries(nominal_parameters, error_parameters)
    identified_errors = diff_dictionaries(current_estimate, error_parameters)
    df_result = result_to_df(simulated_errors, identified_errors)

    exp_handler = ExperimentDataHandler()
    exp_handler.add_figure(fig_error_evolution, 'error_evolution')
    exp_handler.add_figure(fig_param_evolution, 'param_evolution')
    exp_handler.add_note(f"nominal params: \n{RobotDescription.dhparams}\n\n" +
                         f"filtering: {num_obs_filtered}/{num_obs_unfiltered} used\n\n" +
                         f"filter comment: {filter_comment}\n\n "
                         f"parameter identification masks\n{parameter_id_masks}\n\n" +
                         f"error factor\n{factor}\n\n" +
                         f"parameters with errors: \n{error_parameters}\n\n" +
                         f"observations file: \n {observations_file_str_dict[observations_file_select]}\n\n" +
                         f"number of iterations: \n {itervar}/{num_iterations} \n\n " +
                         f"residual_norm_tolerance: \n {residual_norm_tolerance}\n\n" +
                         f"method used \n {method_used}\n\n" +
                         f"jacobian quality\n{jac_quality}\n\n",
                         'info')
    exp_handler.add_note(f"norm residuals evolution \n{norm_residuals_evolution}\n\n", 'norm_convergence')
    exp_handler.add_note(f"{np.sqrt(df_result['identification_accuracy'].pow(2).mean())}", 'residual_error_RMS')
    exp_handler.add_df(df_result, 'data')
    # exp_handler.add_marker_location(list_marker_locations[-1], 'last_marker_locations')
    exp_handler.save_experiment(r'/home/armin/catkin_ws/src/kident2/src/exp')

########################### SETTINGS ###########################
# define which parameters are to be identified
parameter_id_masks = dict()
parameter_id_masks['theta'] =   [False, True, True, True, True, True, True, True, False]
parameter_id_masks['d'] =       [False, True, True, True, True, True, True, True, False]
parameter_id_masks['r'] =       [False, True, True, True, True, True, True, True, False]
parameter_id_masks['alpha'] =   [False, True, True, True, True, True, True, True, False]


# set scaling factor for error
factor = 30

# select observations file
observations_file_select = 1
observations_file_str_dict = {1: r'observation_files/obs_2007_gazebo_iiwa_stopping.bag_20230720-135812.p',
                              5: r'observation_files/obs_bag_with_lockstep_281023_2023-10-28-14-01-49_20231028-142947.p',
                              6: r'observation_files/obs_single_marker_2023-11-01-11-12-21_20231101-112227.p',
                              10: r'observation_files/observations_simulated_w_error_T0_R0_num240_time20231027_113914.p',
                              11: r'observation_files/observations_simulated_w_error_T0.1_R0.1_num240_time20231027_113914.p',
                              12: r'observation_files/observations_simulated_w_error_T1_R1_num240_time20231027_113914.p',
                              13: r'observation_files/observations_simulated_w_error_T2_R2_num240_time20231027_113914.p',
                              14: r'observation_files/observations_simulated_w_error_T5_R5_num240_time20231027_113914.p',
                              15: r'observation_files/observations_simulated_w_error_T10_R10_num240_time20231027_113914.p',
                              16: r'observation_files/observations_simulated_w_error_T20_R20_num240_time20231027_113914.p',
                              17: r'observation_files/observations_simulated_w_error_T30_R30_num240_time20231027_113914.p',
                              20: r'observation_files/obs_single_marker_2023-11-01-11-12-21_20240109-060457.p',
                              21: r'observation_files/observations_simulated_20240109_081340.p'}


# set maximal number of iterations
num_iterations = 12

# tolerance to break loop
residual_norm_tolerance = 1e-3
#################################################################
# for observations_file_select in [1, 4, 9, 10]:
#     for factor in [0, 10, 20, 30, 40, 50, 60, 70]:
#         do_experiment(parameter_id_masks, factor, observations_file_select,
#                       observations_file_str_dict, num_iterations, residual_norm_tolerance)

for observations_file_select in [20]:
    for factor in [10]:
        do_experiment(parameter_id_masks, factor, observations_file_select,
                      observations_file_str_dict, num_iterations, residual_norm_tolerance)