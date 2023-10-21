import datetime
from da_test_suite_functions import *
from exp_data_handler import *

# import nominal parameters
theta_nom = ParameterEstimator.dhparams["theta_nom"].astype(float)
d_nom = ParameterEstimator.dhparams["d_nom"].astype(float)
r_nom = ParameterEstimator.dhparams["r_nom"].astype(float)
alpha_nom = ParameterEstimator.dhparams["alpha_nom"].astype(float)
nominal_parameters = {'theta': theta_nom, 'd': d_nom, 'r': r_nom, 'alpha': alpha_nom}

# define which parameters are to be identified
parameter_id_masks = dict()
parameter_id_masks['theta'] =   [False, True, True, True, True, True, True, True]
parameter_id_masks['d'] =       [False, True, True, True, True, True, True, True]
parameter_id_masks['r'] =       [False, True, True, True, True, True, True, True]
parameter_id_masks['alpha'] =   [False, True, True, True, True, True, True, True]

# apply the errors - model the real robot
factor = 30
theta_error = apply_error_to_params(theta_nom, parameter_id_masks['theta'], factor, 'deg_to_rad')
d_error = apply_error_to_params(d_nom, parameter_id_masks['d'], factor, 'm_to_mm')
r_error = apply_error_to_params(r_nom, parameter_id_masks['r'], factor, 'm_to_mm')
alpha_error = apply_error_to_params(alpha_nom, parameter_id_masks['alpha'], factor, 'deg_to_rad')
error_parameters = {'theta': theta_error, 'd': d_error, 'r': r_error, 'alpha': alpha_error}


# import observations from file
observations_file_str_dict = {1: r'observation_files/obs_2007_gazebo_iiwa_stopping.bag_20230720-135812.p',  # works
                              4: r'observation_files/obs_2007_gazebo_.p',  # works
                              9: r'observation_files/observations_simulated_w_error_0mm_0deg_num24020231020_163148.p',
                              10: r'observation_files/observations_simulated_w_error_0.5mm_0.5deg_num24020231020_164948.p'}
obsservations_file_select = 9
observations_file = open(observations_file_str_dict[obsservations_file_select], 'rb')

observations = pickle.load(observations_file)
observations_file.close()


# number of parameters to identify
total_id_mask = (parameter_id_masks['theta'] + parameter_id_masks['d']
                 + parameter_id_masks['r'] + parameter_id_masks['alpha'])
num_to_ident = sum(bool(x) for x in total_id_mask)  # number of parameters to identify

# initialize list for collecting current marker ids
current_marker = [None]

# initialize list to save marker location as observed
list_marker_locations = list()

# run identification as a loop
current_estimate = error_parameters
current_error_evolution = []
estimated_params_evolution = []
estimated_errors_evolution = []
num_iterations = 8
for i in range(num_iterations):
    print(f"Iteration {i}")
    # below the observation are describing a robot with nominal parameters, and the identification is given
    # the error parameters - this is to be able to vary the error easily
    current_errors, current_estimate, list_marker_locations, jac_quality = identify(observations,
                                                                                    50,
                                                                                    current_estimate,
                                                                                    parameter_id_masks)
    print(f'Jacobian rank: {jac_quality["rank"]}')
    current_error_evolution.append(current_errors)
    estimated_params_evolution.append(current_estimate)
    estimated_errors = diff_dictionaries(current_estimate, nominal_parameters)
    estimated_errors_evolution.append(estimated_errors)


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
identification_accuracy = diff_dictionaries(identified_errors, simulated_errors)
dataframe = result_to_df(diff_dictionaries(nominal_parameters, error_parameters),
                         diff_dictionaries(current_estimate, error_parameters))

exp_handler = ExperimentDataHandler()
exp_handler.add_figure(fig_error_evolution, 'error_evolution')
exp_handler.add_figure(fig_param_evolution, 'param_evolution')
exp_handler.add_note(f"nominal params: \n{ParameterEstimator.dhparams}\n\n" +
                     f"parameter identification masks\n{parameter_id_masks}\n\n" +
                     f"error factor\n{factor}\n\n" +
                     f"parameters with errors: \n{error_parameters}\n\n" +
                     f"observations file: \n {observations_file_str_dict[obsservations_file_select]}\n\n" +
                     f"number of iterations: \n {num_iterations} \n\n", 'settings')
exp_handler.add_df(dataframe, 'data')
exp_handler.add_marker_location(list_marker_locations[-1], 'last_marker_locations')
exp_handler.save_experiment(r'/home/armin/catkin_ws/src/kident2/src/exp')

