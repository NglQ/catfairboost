import os
import pickle
import warnings
import functools
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import equalized_odds_difference
from error_parity import RelaxedThresholdOptimizer
from error_parity.pareto_curve import compute_postprocessing_curve
from error_parity.plotting import plot_postprocessing_frontier
import matplotlib.pyplot as plt

from load_data import get_splits
from params_pipeline import dataset_names, model_names, dataset_name_to_load_fn, dataset_name_to_problem_class, \
    model_name_to_predict_proba_str, model_name_to_training_fn, model_name_to_training_params, \
    model_name_to_colors


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')

# Set seed for consistent results with ExponentiatedGradient
np.random.seed(42)


def compute_pareto_frontier(clf, predict_method_proba_str, tolerance_ticks, data_splits, fit_data, results_filepath,
                            bootstrap, check_cache=True, save_results=True):
    if check_cache and os.path.isfile(results_filepath):
        with open(results_filepath, 'rb') as f:
            return pickle.load(f)

    if fit_data == 'train':
        X_fit, y_fit, s_fit = data_splits['X_train'], data_splits['y_train'], data_splits['s_train']
    else:
        X_fit, y_fit, s_fit = data_splits['X_val'], data_splits['y_val'], data_splits['s_val']

    postproc_results_df = compute_postprocessing_curve(
        model=clf,
        predict_method=predict_method_proba_str,
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


def plot_pareto_frontier(results_df, data_splits, data_type, model_name, color):
    if data_type in ['train', 'val']:
        dtype = 'fit'
    else:
        dtype = 'test'

    plot_postprocessing_frontier(
        results_df,
        perf_metric='accuracy',
        disp_metric='equalized_odds_diff',
        show_data_type=dtype,
        model_name=model_name,
        constant_clf_perf=max((data_splits['y_' + data_type] == const_pred).mean() for const_pred in {0, 1}),
        color=color
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
             acc_eod_results_df,
             hpt_results_filepath=None,
             bootstrap=False,
             tolerance_ticks=None,
             check_model_cache=True,
             training_fn=None,
             training_params=None,
             check_results_cache=True,
             save_results=True,
             colors=('blue', 'green'),
             plot_only_test=True) -> pd.DataFrame:
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
    :param acc_eod_results_df: DataFrame that contains the accuracy and equalized odds difference metrics of previous
        calculated methods. It will be updated during the execution of this function. It saves those metrics for fit
        and test sets, before and after unprocessing. A empty DataFrame is also an acceptable value
    :param hpt_results_filepath: Filepath where the DataFrame, containing the hyperparameter tuning results, is stored.
        If set to None, those results will not be plotted. Default value: None.
    :param bootstrap: Boolean. If set to True, it computes the confidence intervals of the pareto frontier using the
        bootstrap technique. In this case, the results are in terms of mean and std; otherwise, the results will show
        the real scores obtained by the postprocessed model. Default value: False.
    :param tolerance_ticks: List of tolerance floating values of the equalized odds difference. The values should lay
        in the interval [0.0, 1.0]. Default value: None. If None, the postprocessing method will be calculated for the
        following tolerances: [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14,
        0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].
    :param check_model_cache: Boolean. If set to True, the function will try to load the provided model; otherwise,
        it will train the model from scratch. Default value: True.
    :param training_fn: Function responsible for training the base model (i.e., before postprocessing). Pick one from
        the `training.py` module.Default value: None.
    :param training_params: Dictionary containing the base model parameters for training. Default value: None.
    :param check_results_cache: Boolean. If set to True, the function will try to load the provided results; otherwise,
        it will calculate the pareto frontier. Default value: True.
    :param save_results: Boolean. If set to True, it saves the results DataFrame of the pareto frontier. Default value:
        True.
    :param colors: Tuple of two strings. First one is the fit color, second one is the test color. Default value:
        ('blue', 'green').
    :param plot_only_test: Boolean. If set to True, it plots the pareto frontier with respect to the test set only.
        Default value: True.
    :return: A Pandas DataFrame. It returns the updated `acc_eod_results_df`.
    """

    if tolerance_ticks is None or tolerance_ticks == 'default':
        tolerance_ticks = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14,
                           0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
    _curr_acc_eod_res = [acc_fit, eod_fit, acc_test, eod_test]

    fit_color = colors[0]
    test_color = colors[1]

    if hpt_results_filepath is not None:
        assert os.path.isfile(hpt_results_filepath), 'The provided `hpt_results_filepath` does not exist'
        with open(hpt_results_filepath, 'rb') as f:
            hpt_results = pickle.load(f)

        # Filter first 3 models better in accuracy and first 3 better in eod
        best_acc = hpt_results.T.sort_values(by='test_acc', axis=0, ascending=False).T.iloc[:, :3]
        best_eod = hpt_results.T.sort_values(by='test_eod', axis=0, ascending=False).T.iloc[:, :3]
        best_hpt = pd.concat([best_acc, best_eod], axis=1).T.drop_duplicates().T
        best_acc = best_hpt.iloc[0, :]
        best_eod = best_hpt.iloc[1, :]

        plt.scatter(best_acc, best_eod, color=test_color, marker='P', s=65, alpha=0.5, edgecolors='black', label=f'hpt_{model_name}')

    if not plot_only_test:
        plt.scatter([acc_fit], [eod_fit], color=fit_color, marker='^', edgecolors='black', s=140, label=f'{fit_data} results before unprocessing')
    plt.scatter([acc_test], [eod_test], color=test_color, marker='^', s=140, edgecolors='black', label='test results before unprocessing')
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
    acc_eod_results_df[f'{dataset_name}_{model_name}'] = _curr_acc_eod_res + [acc_fit, eod_fit, acc_test, eod_test]

    if not plot_only_test:
        plt.scatter([acc_fit], [eod_fit], color=fit_color, marker='*', edgecolors='black', s=200, label=f'{fit_data} results after unprocessing')
    plt.scatter([acc_test], [eod_test], color=test_color, marker='*', edgecolors='black', s=200, label=f'test results after unprocessing')
    print(f'FIT results after unprocessing: acc: {acc_fit:.4f} - eod: {eod_fit:.4f}')
    print(f'TEST results after unprocessing: acc: {acc_test:.4f} - eod: {eod_test:.4f}')

    results_df = compute_pareto_frontier(model, predict_method_proba_str, tolerance_ticks, data_splits,
                                         fit_data, results_filepath, bootstrap, check_cache=check_results_cache,
                                         save_results=save_results)

    if not plot_only_test:
        plot_pareto_frontier(results_df, data_splits, fit_data, model_name, color=fit_color)
    plot_pareto_frontier(results_df, data_splits, 'test', model_name, color=test_color)

    print('\n---\n')
    return acc_eod_results_df


# Main
if __name__ == '__main__':
    print('Executing main pipeline. If you want to use the results of hyperparameter tuning, run the `hp_tuning` '
          'module first!')

    acc_eod_results_df = pd.DataFrame(
        {'metric': ['before_unpr_acc_fit', 'before_unpr_eod_fit', 'before_unpr_acc_test', 'before_unpr_eod_test',
                    'after_unpr_acc_fit', 'after_unpr_eod_fit', 'after_unpr_acc_test', 'after_unpr_eod_test']}
    )

    for dataset_name, model_name in itertools.product(dataset_names, model_names):
        if model_name == 'lightgbm':
            plt.figure(figsize=(10, 10))
            plt.title(f'Dataset: {dataset_name} - test')

        acc_eod_results_df = pipeline(load_fn=dataset_name_to_load_fn[dataset_name],
                                      problem_class=dataset_name_to_problem_class[dataset_name],
                                      dataset_filepath=f'datasets/{dataset_name}.csv',
                                      fit_data='val',
                                      model_filepath=f'models/{model_name}_val_{dataset_name}.pkl',
                                      model_name=model_name,
                                      predict_method_str='predict',
                                      predict_method_proba_str=model_name_to_predict_proba_str[model_name],
                                      results_filepath=f'results/df_{model_name}_val_{dataset_name}.pkl',
                                      acc_eod_results_df=acc_eod_results_df,
                                      hpt_results_filepath=f'hpt/{dataset_name}_{model_name}.pkl',
                                      bootstrap=False,
                                      tolerance_ticks=None,
                                      check_model_cache=True,
                                      training_fn=model_name_to_training_fn[model_name],
                                      training_params=deepcopy(model_name_to_training_params[model_name]),
                                      check_results_cache=True,
                                      save_results=True,
                                      colors=model_name_to_colors[model_name],
                                      plot_only_test=True)

        if model_name == 'catfairboost':
            plt.xlabel(r"accuracy $\rightarrow$")
            plt.ylabel(r"equalized odds diff tolerance $\leftarrow$")

            # Assuming `dataset_filepath` has the format 'datasets/<dataset_name>.csv'
            plt.savefig(f'plots/val_{dataset_name}.png')
            plt.show()

            # Clear current figure
            plt.clf()

    acc_eod_results_df.set_index('metric', inplace=True)
    with open('acc_eod_results_df.pkl', 'wb') as f:
        pickle.dump(acc_eod_results_df, f)
