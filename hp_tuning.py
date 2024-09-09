import os
import pickle
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import equalized_odds_difference
import lightgbm as lgb
import catboost as cb
from fairgbm import FairGBMClassifier
import optuna
from folktables import ACSIncome, ACSEmployment, ACSIncomePovertyRatio, ACSMobility

from params_models import lightgbm_params, fairgbm_params, catboost_params
from load_data import load_diabetes_easy_cat, load_diabetes_easy_gbm, load_diabetes_hard_cat, load_diabetes_hard_gbm, \
    load_acs_problem_cat, load_acs_problem_gbm, get_splits
from params_pipeline import dataset_names, dataset_name_to_load_fn, dataset_name_to_problem_class


np.random.seed(42)


def hpt_lightgbm(data) -> pd.DataFrame:
    data_splits = get_splits(data)
    results = pd.DataFrame()
    results['metric'] = ['val_acc', 'val_eod']

    def objective(trial):
        enable_bagging = trial.suggest_categorical('enable_bagging', [True, False])
        if enable_bagging:
            bagging_freq = trial.suggest_int('bagging_freq', 1, 30)
        else:
            bagging_freq = 0

        enable_feature_fraction = trial.suggest_categorical('enable_feature_fraction', [True, False])
        if enable_feature_fraction:
            feature_fraction = trial.suggest_float('feature_fraction', 0.5, 0.9)
        else:
            feature_fraction = 1.0

        enable_path_smooth = trial.suggest_categorical('enable_path_smooth', [True, False])
        if enable_path_smooth:
            path_smooth = trial.suggest_float('path_smooth', 1e-5, 0.1, log=True)
        else:
            path_smooth = 0

        max_depth = trial.suggest_int('max_depth', 5, 20)
        num_leaves = trial.suggest_int('num_leaves', 5, 2 ** max_depth)
        if num_leaves > 100_000:
            num_leaves = 100_000

        params = {
            'num_iterations': trial.suggest_int('num_iterations', 20, 300),
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'max_bin': trial.suggest_int('max_bin', 200, 350),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'boosting': trial.suggest_categorical('boosting', ['gbdt', 'dart']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 40),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 0.1, log=True),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.0, 1.0),
            'bagging_freq': bagging_freq,
            'feature_fraction': feature_fraction,
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 0.1, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 0.1, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-5, 0.1, log=True),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'path_smooth': path_smooth
        }

        training_params = deepcopy(lightgbm_params)
        training_params.update(params)

        model = lgb.LGBMClassifier(verbose=0)
        model.set_params(**training_params)

        model.fit(data_splits['X_train'], data_splits['y_train'], categorical_feature=data['cat_cols'])

        y_pred = model.predict(data_splits['X_val'])

        accuracy = accuracy_score(data_splits['y_val'], y_pred)
        eod = equalized_odds_difference(data_splits['y_val'], y_pred, sensitive_features=data_splits['s_val'])
        results[str(params)] = [accuracy, eod]

        print('Accuracy: ', accuracy)
        print('Equalized odds diff: ', eod)

        return accuracy

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=2, show_progress_bar=True)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return results


def hpt_fairgbm(data) -> pd.DataFrame:
    data_splits = get_splits(data)
    results = pd.DataFrame()
    results['metric'] = ['val_acc', 'val_eod']

    def objective(trial):
        enable_bagging = trial.suggest_categorical('enable_bagging', [True, False])
        if enable_bagging:
            bagging_freq = trial.suggest_int('bagging_freq', 1, 30)
        else:
            bagging_freq = 0

        enable_feature_fraction = trial.suggest_categorical('enable_feature_fraction', [True, False])
        if enable_feature_fraction:
            feature_fraction = trial.suggest_float('feature_fraction', 0.5, 0.9)
        else:
            feature_fraction = 1.0

        enable_path_smooth = trial.suggest_categorical('enable_path_smooth', [True, False])
        if enable_path_smooth:
            path_smooth = trial.suggest_float('path_smooth', 1e-5, 0.1, log=True)
        else:
            path_smooth = 0

        max_depth = trial.suggest_int('max_depth', 5, 20)
        num_leaves = trial.suggest_int('num_leaves', 5, 2 ** max_depth)
        if num_leaves > 100_000:
            num_leaves = 100_000

        params = {
            'num_iterations': trial.suggest_int('num_iterations', 20, 300),
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'max_bin': trial.suggest_int('max_bin', 200, 350),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'boosting': trial.suggest_categorical('boosting', ['gbdt', 'dart']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 40),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 0.1, log=True),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.0, 1.0),
            'bagging_freq': bagging_freq,
            'feature_fraction': feature_fraction,
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 0.1, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 0.1, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-5, 0.1, log=True),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'path_smooth': path_smooth,
            'multiplier_learning_rate': trial.suggest_float('multiplier_learning_rate', 1e-5, 1000, log=True)
        }

        training_params = deepcopy(fairgbm_params)
        training_params.update(params)

        model = FairGBMClassifier(
            constraint_type='FNR,FPR',
            **training_params,
        )

        model.fit(data_splits['X_train'], data_splits['y_train'], constraint_group=data_splits['s_train'].to_list(),
                  categorical_feature=data['cat_cols'])

        y_pred = model.predict(data_splits['X_val'])

        accuracy = accuracy_score(data_splits['y_val'], y_pred)
        eod = equalized_odds_difference(data_splits['y_val'], y_pred, sensitive_features=data_splits['s_val'])
        results[str(params)] = [accuracy, eod]

        print('Accuracy: ', accuracy)
        print('Equalized odds diff: ', eod)

        return max(0, accuracy - eod)

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=2, show_progress_bar=True)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return results


def hpt_catboost(data) -> pd.DataFrame:
    data_splits = get_splits(data)
    results = pd.DataFrame()
    results['metric'] = ['val_acc', 'val_eod']

    def objective(trial):
        param = {
            'eval_metric': 'Accuracy',
            'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
            'iterations': trial.suggest_int('iterations', 10, 200),
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1,
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 0.1, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.1, 2.0),
            'depth': trial.suggest_int('depth', 4, 10),
            'boosting_type': 'Plain',
            'bootstrap_type': trial.suggest_categorical(
                'bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']
            )
        }

        if param['bootstrap_type'] == 'Bayesian':
            param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
        elif param['bootstrap_type'] == 'Bernoulli':
            param['subsample'] = trial.suggest_float('subsample', 0.5, 1, log=True)

        model = cb.CatBoostClassifier(**param)
        model.fit(data_splits['X_train'], data_splits['y_train'], cat_features=data['cat_cols'], verbose=0)

        y_pred = model.predict(data_splits['X_val'])

        accuracy = accuracy_score(data_splits['y_val'], y_pred)
        eod = equalized_odds_difference(data_splits['y_val'], y_pred, sensitive_features=data_splits['s_val'])
        results[str(param)] = [accuracy, eod]

        print('Accuracy: ', accuracy)
        print('Equalized odds diff: ', eod)

        return accuracy

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=2, show_progress_bar=True)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return results


if __name__ == '__main__':
    model_names_hpt = ['lightgbm', 'fairgbm', 'catboost']
    model_name_to_hpt_fn = {'lightgbm': hpt_lightgbm,
                            'fairgbm': hpt_fairgbm,
                            'catboost': hpt_catboost}

    os.makedirs('hpt', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)

    print('Hyperparameter tuning of our algos using random search')

    for dataset_name, model_name in itertools.product(dataset_names, model_names_hpt):
        print(f'\n --- Dataset: {dataset_name} - Model: {model_name} ---\n')
        load_fn = dataset_name_to_load_fn[dataset_name]
        problem_class = dataset_name_to_problem_class[dataset_name]
        dataset_filepath = f'datasets/{dataset_name}.csv'
        hpt_fn = model_name_to_hpt_fn[model_name]

        data = load_fn(problem_class, dataset_filepath)
        results = hpt_fn(data)

        with open(f'hpt/{dataset_name}_{model_name}.pkl', 'wb') as f:
            pickle.dump(results, f)

        print('\n---\n')
