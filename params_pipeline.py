from folktables import ACSIncome, ACSEmployment, ACSIncomePovertyRatio, ACSMobility

from load_data import load_diabetes_easy_gbm, load_diabetes_hard_gbm, load_acs_problem_gbm
from training import train_base_lightgbm, train_fair_lightgbm, train_base_catboost, train_fair_catboost, \
    train_base_fairgbm
from params_models import lightgbm_params, fairgbm_params, catboost_params

dataset_names = ['diabetes_easy', 'diabetes_hard', 'acsincome', 'acsemployment', 'acsmobility',
                 'acsincomepovertyratio']

# NOTE: If you change the order of `model_names`, update the if conditions in the main function of `postprocessing.py`
# to have correct plots, if necessary
model_names = ['lightgbm', 'lightfairgbm', 'fairgbm', 'catboost', 'catfairboost']

dataset_name_to_load_fn = {'diabetes_easy': load_diabetes_easy_gbm,
                           'diabetes_hard': load_diabetes_hard_gbm,
                           'acsincome': load_acs_problem_gbm,
                           'acsemployment': load_acs_problem_gbm,
                           'acsmobility': load_acs_problem_gbm,
                           'acsincomepovertyratio': load_acs_problem_gbm}
dataset_name_to_problem_class = {'diabetes_easy': None,
                                 'diabetes_hard': None,
                                 'acsincome': ACSIncome,
                                 'acsemployment': ACSEmployment,
                                 'acsmobility': ACSMobility,
                                 'acsincomepovertyratio': ACSIncomePovertyRatio}
model_name_to_predict_proba_str = {'lightgbm': 'predict_proba',
                                   'lightfairgbm': '_pmf_predict',
                                   'fairgbm': 'predict_proba',
                                   'catboost': 'predict_proba',
                                   'catfairboost': '_pmf_predict'}
model_name_to_training_fn = {'lightgbm': train_base_lightgbm,
                             'lightfairgbm': train_fair_lightgbm,
                             'fairgbm': train_base_fairgbm,
                             'catboost': train_base_catboost,
                             'catfairboost': train_fair_catboost}
model_name_to_training_params = {'lightgbm': lightgbm_params,
                                 'lightfairgbm': lightgbm_params,
                                 'fairgbm': fairgbm_params,
                                 'catboost': catboost_params,
                                 'catfairboost': catboost_params}
model_name_to_colors = {'lightgbm': ('pink', 'green'),
                        'lightfairgbm': ('orange', 'red'),
                        'fairgbm': ('brown', 'blue'),
                        'catboost': ('black', 'purple'),
                        'catfairboost': ('gold', 'cyan')}
