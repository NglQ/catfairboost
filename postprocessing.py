import os
import pickle
import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio
from error_parity import RelaxedThresholdOptimizer
from error_parity.pareto_curve import compute_postprocessing_curve
from error_parity.plotting import plot_postprocessing_frontier
import matplotlib.pyplot as plt
from folktables import ACSIncome, ACSEmployment, ACSIncomePovertyRatio, ACSMobility

from params import lightgbm_params, fairgbm_params
from load_data import load_diabetes_easy_cat, load_diabetes_easy_gbm, load_diabetes_hard_cat, load_diabetes_hard_gbm, \
    load_acs_problem_cat, load_acs_problem_gbm, get_splits
from training import LightFairGBM

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')

# Set seed for consistent results with ExponentiatedGradient
np.random.seed(42)


def compute_pareto_frontier(model, predict_method, tolerance_ticks, data_splits, fit_data, results_filepath, bootstrap,
                            check_cache=True, save_results=True):
    if check_cache and os.path.isfile(results_filepath):
        with open(results_filepath, 'rb') as f:
            return pickle.load(f)

    if fit_data == 'train':
        X_fit, y_fit, s_fit = data_splits['X_train'], data_splits['y_train'], data_splits['s_train']
    else:
        X_fit, y_fit, s_fit = data_splits['X_val'], data_splits['y_val'], data_splits['s_val']

    postproc_results_df = compute_postprocessing_curve(
        model=model,
        predict_method=predict_method,
        fit_data=(X_fit, y_fit, s_fit),
        eval_data={'test': (data_splits['X_test'], data_splits['y_test'], data_splits['s_test'])},
        fairness_constraint='equalized_odds',
        tolerance_ticks=tolerance_ticks,
        bootstrap=bootstrap,
        seed=42
    )

    if save_results:
        with open(results_filepath, 'wb') as f:
            pickle.dump(postproc_results_df, f)

    return postproc_results_df


def plot_pareto_frontier(results_df, data_splits, data_type, model_name, img_size):
    plt.figure(figsize=img_size)

    plot_postprocessing_frontier(
        results_df,
        perf_metric='accuracy',
        disp_metric='equalized_odds_diff',
        show_data_type=data_type,
        model_name=model_name,
        constant_clf_perf=max((data_splits['y_' + data_type] == const_pred).mean() for const_pred in {0, 1}),
    )


def eval_acc_and_eod(predict_method, X, y_true, s):
    y_pred = predict_method(X)

    acc = accuracy_score(y_true, y_pred)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=s)

    return acc, eod


def pipeline(load_fn,
             problem_class,
             dataset_filepath,
             fit_data,  # can only be 'train' or 'val'
             model_filepath,
             model_name,
             predict_method,
             predict_method_str,
             results_filepath,
             bootstrap,
             tolerance_ticks=None,
             check_model_cache=True,
             training_fn=None,
             training_params=None,
             check_results_cache=True,
             save_results=True,
             img_size=(10, 10)):

    if tolerance_ticks is None or tolerance_ticks == 'default':
        tolerance_ticks = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14,
                           0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        assert isinstance(tolerance_ticks, list)

    assert fit_data in ['train', 'val']
    assert dataset_filepath.startswith('datasets/') and dataset_filepath.endswith('.csv'), \
        '`dataset_filepath` must have the format `datasets/<dataset_name>.csv`'

    # Create the necessary directories if they do not exist
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    data = load_fn(problem_class, dataset_filepath)
    data_splits = get_splits(data)

    # Set seed for consistent results with ExponentiatedGradient
    np.random.seed(42)

    if check_model_cache and os.path.isfile(model_filepath):
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)
    else:
        assert training_params is not None and training_fn is not None
        print(f'Training {model_name}...')
        model, _data, _y_pred, _training_time = training_fn(data, data_splits, training_params)

    # Before unprocessing
    acc_fit, eod_fit = eval_acc_and_eod(predict_method, data_splits['X_' + fit_data], data_splits['y_' + fit_data],
                                        data_splits['s_' + fit_data])
    acc_test, eod_test = eval_acc_and_eod(predict_method, data_splits['X_test'], data_splits['y_test'],
                                          data_splits['s_test'])
    plt.scatter([acc_fit], [eod_fit], color='blue', marker='+', s=20, label=f'{fit_data} results before unprocessing')
    plt.scatter([acc_test], [eod_test], color='green', marker='+', s=20, label='test results before unprocessing')
    print(f'FIT results before unprocessing: acc: {acc_fit:.4f} - eod: {eod_fit:.4f}')
    print(f'TEST results before unprocessing: acc: {acc_test:.4f} - eod: {eod_test:.4f}')

    results_df = compute_pareto_frontier(model, predict_method_str, tolerance_ticks, data_splits, fit_data,
                                         results_filepath, bootstrap, check_cache=check_results_cache,
                                         save_results=save_results)

    # After unprocessing: model with tolerance=1.0
    row_idx = results_df['tolerance'][results_df['tolerance'] == 1.0].index[0]
    acc_fit = results_df[f'accuracy_{'mean_' if bootstrap else ''}fit'][row_idx]
    acc_test = results_df[f'accuracy_{'mean_' if bootstrap else ''}test'][row_idx]
    eod_fit = results_df[f'equalized_odds_diff_{'mean_' if bootstrap else ''}fit'][row_idx]
    eod_test = results_df[f'equalized_odds_diff_{'mean_' if bootstrap else ''}test'][row_idx]
    plt.scatter([acc_fit], [eod_fit], color='blue', marker='*', s=20,
                label=f'{fit_data} results after unprocessing{' with bootstrap' if bootstrap else ''}')
    plt.scatter([acc_test], [eod_test], color='green', marker='*', s=20,
                label=f'test results after unprocessing{' with bootstrap' if bootstrap else ''}')
    print(f'FIT results before unprocessing{' with bootstrap' if bootstrap else ''}: '
          f'acc: {acc_fit:.4f} - eod: {eod_fit:.4f}')
    print(f'TEST results before unprocessing{' with bootstrap' if bootstrap else ''}: '
          f'acc: {acc_test:.4f} - eod: {eod_test:.4f}')

    plot_pareto_frontier(results_df, data_splits, fit_data, model_name, img_size)
    plot_pareto_frontier(results_df, data_splits, 'test', model_name, img_size)

    plt.xlabel(r"accuracy $\rightarrow$")
    plt.ylabel(r"equalized odds diff tolerance $\leftarrow$")

    # Assuming `dataset_filepath` has the format 'datasets/<dataset_name>.csv', see above asserts
    plt.savefig(f'plots/{model_name}_{fit_data}_{dataset_filepath.split('/')[1].split('.')[0]}.png')
    plt.show()

    plt.clf()


# -----

data = load_acs_problem_gbm(ACSIncome, 'datasets/acsincome.csv')
data = get_splits(data)


with open('models/lightfairgbm_acsincome.pkl', 'rb') as f:
    model = pickle.load(f)

