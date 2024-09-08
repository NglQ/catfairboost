import os
import pickle
import warnings
import functools

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio
from error_parity import RelaxedThresholdOptimizer
from error_parity.pareto_curve import compute_postprocessing_curve
from error_parity.plotting import plot_postprocessing_frontier
import matplotlib.pyplot as plt
from folktables import ACSIncome, ACSEmployment, ACSIncomePovertyRatio, ACSMobility

from params import lightgbm_params, fairgbm_params, catboost_params
from load_data import load_diabetes_easy_cat, load_diabetes_easy_gbm, load_diabetes_hard_cat, load_diabetes_hard_gbm, \
    load_acs_problem_cat, load_acs_problem_gbm, get_splits
from training import LightFairGBM, train_base_lightgbm, train_fair_lightgbm, train_base_catboost, train_fair_catboost, \
    train_base_fairgbm

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')

# Set seed for consistent results with ExponentiatedGradient
np.random.seed(42)


def compute_pareto_frontier(unprocessed_clf, tolerance_ticks, data_splits, fit_data, results_filepath, bootstrap,
                            check_cache=True, save_results=True):
    if check_cache and os.path.isfile(results_filepath):
        with open(results_filepath, 'rb') as f:
            return pickle.load(f)

    data_splits['X_train'].attrs['type'] = 'train'
    data_splits['X_val'].attrs['type'] = 'val'
    data_splits['X_test'].attrs['type'] = 'test'

    class RelaxedThresholdOptimizerParetoComputable:
        def __init__(self, plain_trained_relaxed_threshold_optimizer):
            self.model = plain_trained_relaxed_threshold_optimizer
            self.fit_rows = data_splits['X_' + fit_data].head(5)

        def predict(self, X) -> np.ndarray:
            # TODO: is shuffling happening?

            X_rows = X.head(5)
            if X_rows.equals(self.fit_rows):
                group_label = 's_' + fit_data
            else:
                group_label = 's_test'

            return self.model(X, group=data_splits[group_label])

    # unprocessed_clf = RelaxedThresholdOptimizerParetoComputable(unprocessed_clf)

    if fit_data == 'train':
        X_fit, y_fit, s_fit = data_splits['X_train'], data_splits['y_train'], data_splits['s_train']
    else:
        X_fit, y_fit, s_fit = data_splits['X_val'], data_splits['y_val'], data_splits['s_val']

    postproc_results_df = compute_postprocessing_curve(
        model=unprocessed_clf,
        predict_method='predict',
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


def plot_pareto_frontier(results_df, data_splits, data_type, model_name):
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
             predict_method_str,
             predict_method_proba_str,
             results_filepath,
             bootstrap,
             tolerance_ticks=None,
             check_model_cache=True,
             training_fn=None,
             training_params=None,
             check_results_cache=True,
             save_results=True,
             img_size=(10, 10)) -> None:
    """
    Main function of our project.
    First, it trains a specific model by using the `training_fn` parameter, then it computes the pareto frontier of said
    model by putting in relation the accuracy with the equalized odds difference.
    To do that, we leverage the `error-parity` library.
    Finally, this function prints, plots and saves the main results.

    :param load_fn: Function to load the dataset. Pick one from the `load_data.py` module.
    :param problem_class: Dataset type of the `folktables` library. Use this parameter if you want to use a folktable
        dataset.
    :param dataset_filepath: Filepath of the dataset to load. If the file does not exist, the function will create and
        save the dataset at this location.
    :param fit_data: A string. The only accepted values are `train` and `val`. It is used for selecting the data split
        that will be used to train the postprocessing fairness technique.
    :param model_filepath: Filepath where the function tries to load the model. If the file does not exist, the function
        will use the `training_fn` parameter to train the model and save it at this location.
    :param model_name: A string. Name of the model. Used for plotting and naming files.
    :param predict_method_str: A string. Exact name of the model class method for predicting the labels.
    :param predict_method_proba_str: A string. Exact name of the model class method for predicting the labels'
        probabilities.
    :param results_filepath: Filepath where the DataFrame containing the pareto frontier results will be saved.
    :param bootstrap: Boolean. If set to True, it computes the confidence intervals of the pareto frontier using the
        bootstrap technique. In this case, the results are in terms of mean and std; otherwise, the results will show
        the real scores obtained by the postprocessed model.
    :param tolerance_ticks: List of tolerance floating values of the equalized odds difference. The values should lay
        in the interval [0.0, 1.0]. Default value: None. If None, the postprocessing method will be calculated for the
        following tolerances: [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14,
        0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
    :param check_model_cache: Boolean. If set to True, the function will try to load the provided model; otherwise,
        it will train the model from scratch. Default value: True.
    :param training_fn: Function responsible for training the base model (i.e., before postprocessing). Pick one from
        the `training.py` module.Default value: None.
    :param training_params: Dictionary containing the base model parameters for training. Default value: None.
    :param check_results_cache: Boolean. If set to True, the function will try to load the provided results; otherwise,
        it will calculate the pareto frontier. Default value: True.
    :param save_results: Boolean. If set to True, it saves the results DataFrame of the pareto frontier. Default value:
        True.
    :param img_size: Tuple of two integers. Size of the plots.
    :return: None. This function does not return any data.
    """

    if tolerance_ticks is None or tolerance_ticks == 'default':
        tolerance_ticks = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14,
                           0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        assert isinstance(tolerance_ticks, list)

    assert fit_data in ['train', 'val']
    assert dataset_filepath.startswith('datasets/') and dataset_filepath.endswith('.csv'), \
        '`dataset_filepath` must have the format `datasets/<dataset_name>.csv`'

    # Create the necessary directories if they do not exist
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    data = load_fn(problem_class, dataset_filepath)
    data_splits = get_splits(data)

    # Set seed for consistent results with ExponentiatedGradient
    np.random.seed(42)

    print('\n---\n')
    dataset_name = dataset_filepath.split('/')[1].split('.')[0]
    print(f'Model name: {model_name} - Dataset: {dataset_name}')

    if check_model_cache and os.path.isfile(model_filepath):
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)
    else:
        assert training_params is not None and training_fn is not None, ('If no model can be loaded, you need to '
                                                                         'provide the `training_params` and '
                                                                         '`training_fn` parameters.')
        print(f'Training {model_name}...')
        model, _data, _y_pred, _training_time = training_fn(data, data_splits, training_params)

        if save_results:
            with open(model_filepath, 'wb') as f:
                pickle.dump(model, f)

    # Before unprocessing
    _predict = getattr(model, predict_method_str)
    acc_fit, eod_fit = eval_acc_and_eod(_predict, data_splits['X_' + fit_data], data_splits['y_' + fit_data],
                                        data_splits['s_' + fit_data])
    acc_test, eod_test = eval_acc_and_eod(_predict, data_splits['X_test'], data_splits['y_test'],
                                          data_splits['s_test'])
    plt.figure(figsize=img_size)
    plt.scatter([acc_fit], [eod_fit], color='blue', marker='+', s=20, label=f'{fit_data} results before unprocessing')
    plt.scatter([acc_test], [eod_test], color='green', marker='+', s=20, label='test results before unprocessing')
    print(f'FIT results before unprocessing: acc: {acc_fit:.4f} - eod: {eod_fit:.4f}')
    print(f'TEST results before unprocessing: acc: {acc_test:.4f} - eod: {eod_test:.4f}')

    _predict_proba_before_unprocessing = getattr(model, predict_method_proba_str)
    unprocessed_clf = RelaxedThresholdOptimizer(
        predictor=lambda X: _predict_proba_before_unprocessing(X)[:, -1],
        constraint='equalized_odds',
        tolerance=1.0
    )
    unprocessed_clf.fit(X=data_splits['X_' + fit_data], y=data_splits['y_' + fit_data], group=data_splits['s_' + fit_data])

    # After unprocessing
    _predict_fit_after_unprocessing = functools.partial(unprocessed_clf.predict, group=data_splits['s_' + fit_data])
    _predict_test_after_unprocessing = functools.partial(unprocessed_clf.predict, group=data_splits['s_test'])

    acc_fit, eod_fit = eval_acc_and_eod(_predict_fit_after_unprocessing, data_splits['X_' + fit_data],
                                        data_splits['y_' + fit_data], data_splits['s_' + fit_data])
    acc_test, eod_test = eval_acc_and_eod(_predict_test_after_unprocessing, data_splits['X_test'], data_splits['y_test'],
                                          data_splits['s_test'])
    plt.scatter([acc_fit], [eod_fit], color='blue', marker='*', s=20, label=f'{fit_data} results after unprocessing')
    plt.scatter([acc_test], [eod_test], color='green', marker='*', s=20, label=f'test results after unprocessing')
    print(f'FIT results after unprocessing: acc: {acc_fit:.4f} - eod: {eod_fit:.4f}')
    print(f'TEST results after unprocessing: acc: {acc_test:.4f} - eod: {eod_test:.4f}')

    results_df = compute_pareto_frontier(unprocessed_clf, tolerance_ticks, data_splits, fit_data,
                                         results_filepath, bootstrap, check_cache=check_results_cache,
                                         save_results=save_results)

    plot_pareto_frontier(results_df, data_splits, fit_data, model_name)
    plot_pareto_frontier(results_df, data_splits, 'test', model_name)

    plt.xlabel(r"accuracy $\rightarrow$")
    plt.ylabel(r"equalized odds diff tolerance $\leftarrow$")

    # Assuming `dataset_filepath` has the format 'datasets/<dataset_name>.csv', see above asserts
    plt.savefig(f'plots/{model_name}_{fit_data}_{dataset_name}.png')
    plt.show()

    # Clear current figure
    plt.clf()

    print('\n---\n')


# Main
if __name__ == '__main__':
    pipeline(load_diabetes_easy_gbm,
             None,
             'datasets/diabetes_easy.csv',
             'train',
             'models/lightgbm_diabetes_easy.pkl',
             'lightgbm',
             'predict',
             'predict_proba',
             'results/df_lightgbm_diabetes_easy.pkl',
             False,
             tolerance_ticks=[0.1],
             check_model_cache=True,
             training_fn=train_base_lightgbm,
             training_params=lightgbm_params,
             check_results_cache=True,
             save_results=True,
             img_size=(10, 10))
