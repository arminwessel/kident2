import pickle
import numpy as np
import matplotlib.pyplot as plt
from parameter_estimator import ParameterEstimator
import utils
from itertools import combinations
import random
import time
from scipy.optimize import least_squares
from da_test_suite_functions import *

# import nominal parameters
theta_nom = ParameterEstimator.dhparams["theta_nom"].astype(float)
d_nom = ParameterEstimator.dhparams["d_nom"].astype(float)
r_nom = ParameterEstimator.dhparams["r_nom"].astype(float)
alpha_nom = ParameterEstimator.dhparams["alpha_nom"].astype(float)
nominal_parameters = {'theta': theta_nom, 'd': d_nom, 'r': r_nom, 'alpha': alpha_nom}

# define which parameters are to be identified
parameter_id_masks = dict()
parameter_id_masks['theta'] = [False, True, True, True, True, True, False, False]
parameter_id_masks['d'] = [False, True, True, True, True, True, False, False]
parameter_id_masks['r'] = [False, True, True, True, True, True, False, False]
parameter_id_masks['alpha'] = [False, True, True, True, True, True, False, False]

# apply the errors - model the real robot
theta_error = apply_error_to_params(theta_nom, parameter_id_masks['theta'], 0, 'deg_to_rad')
d_error = apply_error_to_params(d_nom, parameter_id_masks['d'], 0, 'm_to_mm')
r_error = apply_error_to_params(r_nom, parameter_id_masks['r'], 0, 'm_to_mm')
alpha_error = apply_error_to_params(alpha_nom, parameter_id_masks['alpha'], 0, 'deg_to_rad')
error_parameters = {'theta': theta_error, 'd': d_error, 'r': r_error, 'alpha': alpha_error}


# import observations from file
observations_file_str_list = ['obs_2007_gazebo_iiwa_stopping.bag_20230720-135812.p',
                              'observations_fake.p',
                              'observations_fake_neu.p',
                              'obs_2007_gazebo_.p',
                              'observations_simulated_20231010_074809.p']
observations_file = open(observations_file_str_list[4], 'rb')
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
for i in range(2):
    print(f"Iteration {i}")
    # below the observation are describing a robot with nominal parameters, and the identification is given
    # the error parameters - this is to be able to vary the error easily
    current_errors, estimated_params, list_marker_locations = identify(observations, 50, current_estimate, parameter_id_masks)
    current_estimate = estimated_params
    current_error_evolution.append(current_errors)
    estimated_params_evolution.append(current_estimate)
    estimated_errors_evolution.append(diff_dictionaries(current_estimate, nominal_parameters))

titles = {'suptitle': 'Evolution of current estimate',
          'theta': r'$\theta$ [°]',
          'd': r'd [mm]',
          'r': r'r [mm]',
          'alpha': r'$\alpha$ [°]'}
plot_evolution(titles, estimated_params_evolution)

titles = {'suptitle': 'Evolution of parameter errors',
          'theta': r'$\Delta$$\theta$ [°]',
          'd': r'$\Delta$d [mm]',
          'r': r'$\Delta$r [mm]',
          'alpha': r'$\Delta$$\alpha$ [°]'}
plot_evolution(titles, estimated_errors_evolution)

df = result_to_df(error_parameters, current_estimate)
dataframe_to_pdf(df, 'test_1.pdf')
with open('marker_locations.p', 'wb') as f:  # open a text file
    pickle.dump(list_marker_locations, f) # serialize the list

plt.show()

print('fin')