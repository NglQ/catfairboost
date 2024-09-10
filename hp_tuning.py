import os
import threading
import signal
import pickle
import itertools
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
import lightgbm as lgb
import catboost as cb
from fairgbm import FairGBMClassifier
from error_parity import RelaxedThresholdOptimizer
import optuna
from termcolor import colored

from params_models import lightgbm_params, fairgbm_params
from load_data import get_splits
from training import LightFairGBM
from params_pipeline import dataset_names, model_names, dataset_name_to_load_fn, dataset_name_to_problem_class

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='Using categorical_feature in Dataset.')

np.random.seed(42)


def stop_trial(trial, ppid):
    trial.should_prune = True
    print(colored(f'Stopping trial {trial.number}...', 'red'))
    os.kill(ppid, signal.SIGUSR1)


def signal_handler(signum, frame):
    raise TimeoutError(colored('STOPPED, time budget expired!', 'red'))


def hpt_lightgbm(data) -> tuple[pd.DataFrame, lgb.LGBMClassifier]:
    data_splits = get_splits(data)
    results = pd.DataFrame()
    results['metric'] = ['test_acc', 'test_eod', 'val_unpr_acc', 'val_unpr_eod']
    best_model: lgb.LGBMClassifier = None
    best_val_unpr_acc = float('-inf')

    def objective(trial):
        nonlocal best_model, best_val_unpr_acc

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

        model = lgb.LGBMClassifier(verbosity=-1)
        model.set_params(**training_params)

        model.fit(data_splits['X_train'], data_splits['y_train'], categorical_feature=data['cat_cols'])

        y_pred = model.predict(data_splits['X_val'])

        val_accuracy = accuracy_score(data_splits['y_val'], y_pred)

        # --- test results not used in the hyperparameter tuning ---
        y_pred = model.predict(data_splits['X_test'])
        test_accuracy = accuracy_score(data_splits['y_test'], y_pred)
        test_eod = equalized_odds_difference(data_splits['y_test'], y_pred, sensitive_features=data_splits['s_test'])

        # --- unprocess model
        unprocessed_clf = RelaxedThresholdOptimizer(
            predictor=lambda X: model.predict_proba(X)[:, -1],
            constraint='equalized_odds',
            tolerance=1.0
        )
        unprocessed_clf.fit(X=data_splits['X_val'], y=data_splits['y_val'], group=data_splits['s_val'])
        y_pred = unprocessed_clf.predict(data_splits['X_val'], group=data_splits['s_val'])
        val_unpr_accuracy = accuracy_score(data_splits['y_val'], y_pred)
        val_unpr_eod = equalized_odds_difference(data_splits['y_val'], y_pred, sensitive_features=data_splits['s_val'])
        results[str(params)] = [test_accuracy, test_eod, val_unpr_accuracy, val_unpr_eod]
        if val_unpr_accuracy > best_val_unpr_acc:
            print(colored('Found a better unprocessed classifier!', 'green'))
            best_val_unpr_acc = val_unpr_accuracy
            best_model = model

        print(colored(f'Test accuracy: {test_accuracy}', 'magenta'))
        print(colored(f'Test equalized odds diff: {test_eod}', 'magenta'))
        print(colored(f'Validation unprocessed accuracy: {val_unpr_accuracy}', 'magenta'))
        print(colored(f'Validation unprocessed equalized odds diff: {val_unpr_eod}', 'magenta'))

        return val_accuracy

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(colored(f'Number of finished trials: {len(study.trials)}', 'cyan'))
    print(colored(f'Best trial: {study.best_trial.params}', 'cyan'))
    return results, best_model


def hpt_lightfairgbm(data) -> tuple[pd.DataFrame, ExponentiatedGradient]:
    data_splits = get_splits(data)
    results = pd.DataFrame()
    results['metric'] = ['test_acc', 'test_eod', 'val_unpr_acc', 'val_unpr_eod']
    best_model: ExponentiatedGradient = None
    best_val_unpr_acc = float('-inf')

    def objective(trial):
        nonlocal best_model, best_val_unpr_acc

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

        internal_model = LightFairGBM(verbosity=-1)
        training_params.update({'cat_cols': data['cat_cols']})
        internal_model.set_params(**training_params)

        model = ExponentiatedGradient(
            estimator=internal_model,
            constraints=EqualizedOdds()
        )

        signal.signal(signal.SIGUSR1, signal_handler)
        timer = threading.Timer(time_budget, lambda: stop_trial(trial, os.getpid()))
        timer.start()
        val_accuracy = 0.0
        try:
            model.fit(data_splits['X_train'], data_splits['y_train'], sensitive_features=data_splits['s_train'])

            y_pred = model.predict(data_splits['X_val'])

            val_accuracy = accuracy_score(data_splits['y_val'], y_pred)

            # --- test results not used in the hyperparameter tuning ---
            y_pred = model.predict(data_splits['X_test'])
            test_accuracy = accuracy_score(data_splits['y_test'], y_pred)
            test_eod = equalized_odds_difference(data_splits['y_test'], y_pred, sensitive_features=data_splits['s_test'])

            # --- unprocess model
            unprocessed_clf = RelaxedThresholdOptimizer(
                predictor=lambda X: model._pmf_predict(X)[:, -1],
                constraint='equalized_odds',
                tolerance=1.0
            )
            unprocessed_clf.fit(X=data_splits['X_val'], y=data_splits['y_val'], group=data_splits['s_val'])
            y_pred = unprocessed_clf.predict(data_splits['X_val'], group=data_splits['s_val'])
            val_unpr_accuracy = accuracy_score(data_splits['y_val'], y_pred)
            val_unpr_eod = equalized_odds_difference(data_splits['y_val'], y_pred, sensitive_features=data_splits['s_val'])
            results[str(params)] = [test_accuracy, test_eod, val_unpr_accuracy, val_unpr_eod]
            if val_unpr_accuracy > best_val_unpr_acc:
                print(colored('Found a better unprocessed classifier!', 'green'))
                best_val_unpr_acc = val_unpr_accuracy
                best_model = model

            print(colored(f'Test accuracy: {test_accuracy}', 'magenta'))
            print(colored(f'Test equalized odds diff: {test_eod}', 'magenta'))
            print(colored(f'Validation unprocessed accuracy: {val_unpr_accuracy}', 'magenta'))
            print(colored(f'Validation unprocessed equalized odds diff: {val_unpr_eod}', 'magenta'))

        except TimeoutError as err:
            print(err)

        finally:
            timer.cancel()

        return val_accuracy

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(colored(f'Number of finished trials: {len(study.trials)}', 'cyan'))
    print(colored(f'Best trial: {study.best_trial.params}', 'cyan'))
    return results, best_model


def hpt_fairgbm(data) -> tuple[pd.DataFrame, FairGBMClassifier]:
    data_splits = get_splits(data)
    results = pd.DataFrame()
    results['metric'] = ['test_acc', 'test_eod', 'val_unpr_acc', 'val_unpr_eod']
    best_model: FairGBMClassifier = None
    best_val_unpr_acc = float('-inf')

    def objective(trial):
        nonlocal best_model, best_val_unpr_acc

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

        val_accuracy = accuracy_score(data_splits['y_val'], y_pred)
        val_eod = equalized_odds_difference(data_splits['y_val'], y_pred, sensitive_features=data_splits['s_val'])

        # --- test results not used in the hyperparameter tuning ---
        y_pred = model.predict(data_splits['X_test'])
        test_accuracy = accuracy_score(data_splits['y_test'], y_pred)
        test_eod = equalized_odds_difference(data_splits['y_test'], y_pred, sensitive_features=data_splits['s_test'])

        # --- unprocess model
        unprocessed_clf = RelaxedThresholdOptimizer(
            predictor=lambda X: model.predict_proba(X)[:, -1],
            constraint='equalized_odds',
            tolerance=1.0
        )
        unprocessed_clf.fit(X=data_splits['X_val'], y=data_splits['y_val'], group=data_splits['s_val'])
        y_pred = unprocessed_clf.predict(data_splits['X_val'], group=data_splits['s_val'])
        val_unpr_accuracy = accuracy_score(data_splits['y_val'], y_pred)
        val_unpr_eod = equalized_odds_difference(data_splits['y_val'], y_pred, sensitive_features=data_splits['s_val'])
        results[str(params)] = [test_accuracy, test_eod, val_unpr_accuracy, val_unpr_eod]
        if val_unpr_accuracy > best_val_unpr_acc:
            print(colored('Found a better unprocessed classifier!', 'green'))
            best_val_unpr_acc = val_unpr_accuracy
            best_model = model

        print(colored(f'Test accuracy: {test_accuracy}', 'magenta'))
        print(colored(f'Test equalized odds diff: {test_eod}', 'magenta'))
        print(colored(f'Validation unprocessed accuracy: {val_unpr_accuracy}', 'magenta'))
        print(colored(f'Validation unprocessed equalized odds diff: {val_unpr_eod}', 'magenta'))

        return max(0, val_accuracy - val_eod)

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(colored(f'Number of finished trials: {len(study.trials)}', 'cyan'))
    print(colored(f'Best trial: {study.best_trial.params}', 'cyan'))
    return results, best_model


def hpt_catboost(data) -> tuple[pd.DataFrame, cb.CatBoostClassifier]:
    data_splits = get_splits(data)
    results = pd.DataFrame()
    results['metric'] = ['test_acc', 'test_eod', 'val_unpr_acc', 'val_unpr_eod']
    best_model: lgb.LGBMClassifier = None
    best_val_unpr_acc = float('-inf')

    def objective(trial):
        nonlocal best_model, best_val_unpr_acc

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

        val_accuracy = accuracy_score(data_splits['y_val'], y_pred)

        # --- test results not used in the hyperparameter tuning ---
        y_pred = model.predict(data_splits['X_test'])
        test_accuracy = accuracy_score(data_splits['y_test'], y_pred)
        test_eod = equalized_odds_difference(data_splits['y_test'], y_pred, sensitive_features=data_splits['s_test'])

        # --- unprocess model
        unprocessed_clf = RelaxedThresholdOptimizer(
            predictor=lambda X: model.predict_proba(X)[:, -1],
            constraint='equalized_odds',
            tolerance=1.0
        )
        unprocessed_clf.fit(X=data_splits['X_val'], y=data_splits['y_val'], group=data_splits['s_val'])
        y_pred = unprocessed_clf.predict(data_splits['X_val'], group=data_splits['s_val'])
        val_unpr_accuracy = accuracy_score(data_splits['y_val'], y_pred)
        val_unpr_eod = equalized_odds_difference(data_splits['y_val'], y_pred, sensitive_features=data_splits['s_val'])
        results[str(param)] = [test_accuracy, test_eod, val_unpr_accuracy, val_unpr_eod]
        if val_unpr_accuracy > best_val_unpr_acc:
            print(colored('Found a better unprocessed classifier!', 'green'))
            best_val_unpr_acc = val_unpr_accuracy
            best_model = model

        print(colored(f'Test accuracy: {test_accuracy}', 'magenta'))
        print(colored(f'Test equalized odds diff: {test_eod}', 'magenta'))
        print(colored(f'Validation unprocessed accuracy: {val_unpr_accuracy}', 'magenta'))
        print(colored(f'Validation unprocessed equalized odds diff: {val_unpr_eod}', 'magenta'))

        return val_accuracy

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(colored(f'Number of finished trials: {len(study.trials)}', 'cyan'))
    print(colored(f'Best trial: {study.best_trial.params}', 'cyan'))
    return results, best_model


def hpt_catfairboost(data) -> tuple[pd.DataFrame, ExponentiatedGradient]:
    data_splits = get_splits(data)
    results = pd.DataFrame()
    results['metric'] = ['test_acc', 'test_eod', 'val_unpr_acc', 'val_unpr_eod']
    best_model: ExponentiatedGradient = None
    best_val_unpr_acc = float('-inf')

    def objective(trial):
        nonlocal best_model, best_val_unpr_acc

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

        param.update({'cat_features': data['cat_cols']})

        # In `training_params` kwargs dict expecting value `cat_features`
        assert 'cat_features' in param, 'Expected `cat_features` in training_params'
        internal_model = cb.CatBoostClassifier(**param)

        model = ExponentiatedGradient(
            estimator=internal_model,
            constraints=EqualizedOdds()
        )

        signal.signal(signal.SIGUSR1, signal_handler)
        timer = threading.Timer(time_budget, lambda: stop_trial(trial, os.getpid()))
        timer.start()
        val_accuracy = 0.0
        try:
            model.fit(data_splits['X_train'], data_splits['y_train'], sensitive_features=data_splits['s_train'])

            y_pred = model.predict(data_splits['X_val'])

            val_accuracy = accuracy_score(data_splits['y_val'], y_pred)

            # --- test results not used in the hyperparameter tuning ---
            y_pred = model.predict(data_splits['X_test'])
            test_accuracy = accuracy_score(data_splits['y_test'], y_pred)
            test_eod = equalized_odds_difference(data_splits['y_test'], y_pred, sensitive_features=data_splits['s_test'])

            # --- unprocess model
            unprocessed_clf = RelaxedThresholdOptimizer(
                predictor=lambda X: model._pmf_predict(X)[:, -1],
                constraint='equalized_odds',
                tolerance=1.0
            )
            unprocessed_clf.fit(X=data_splits['X_val'], y=data_splits['y_val'], group=data_splits['s_val'])
            y_pred = unprocessed_clf.predict(data_splits['X_val'], group=data_splits['s_val'])
            val_unpr_accuracy = accuracy_score(data_splits['y_val'], y_pred)
            val_unpr_eod = equalized_odds_difference(data_splits['y_val'], y_pred, sensitive_features=data_splits['s_val'])
            results[str(param)] = [test_accuracy, test_eod, val_unpr_accuracy, val_unpr_eod]
            if val_unpr_accuracy > best_val_unpr_acc:
                print(colored('Found a better unprocessed classifier!', 'green'))
                best_val_unpr_acc = val_unpr_accuracy
                best_model = model

            print(colored(f'Test accuracy: {test_accuracy}', 'magenta'))
            print(colored(f'Test equalized odds diff: {test_eod}', 'magenta'))
            print(colored(f'Validation unprocessed accuracy: {val_unpr_accuracy}', 'magenta'))
            print(colored(f'Validation unprocessed equalized odds diff: {val_unpr_eod}', 'magenta'))

        # It's necessary to include the `KeyboardInterrupt` exception for CatFairBoost. This is because CatBoost,
        # in its internal libraries, could convert the TimeoutError into a KeyboardInterrupt exception.
        # Downsides: the code is not stoppable with CTRL+C anymore inside CatFairBoost hyperparameter tuning.
        except (TimeoutError, KeyboardInterrupt) as err:
            print(err)

        finally:
            timer.cancel()

        return val_accuracy

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(colored(f'Number of finished trials: {len(study.trials)}', 'cyan'))
    print(colored(f'Best trial: {study.best_trial.params}', 'cyan'))
    return results, best_model


if __name__ == '__main__':
    # 7 minutes time budget for lightfairgbm and catfairboost; otherwise, infinite time
    time_budget = 7 * 60

    model_name_to_hpt_fn = {'lightgbm': hpt_lightgbm,
                            'lightfairgbm': hpt_lightfairgbm,
                            'fairgbm': hpt_fairgbm,
                            'catboost': hpt_catboost,
                            'catfairboost': hpt_catfairboost}

    os.makedirs('hpt', exist_ok=True)
    os.makedirs('hpt_best_models', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)

    print(colored('Hyperparameter tuning of our selected algorithms using random search.', 'green',
                  attrs=['bold']))
    print(colored('If you want to stop the code, it could happen that CTRL+C does not work (the feature has been '
                  'disabled during CatFairBoost tuning because of the inner workings of the CatBoost library); '
                  'if that is the case, you need to send a SIGTERM(15) signal or close the terminal.','red'))

    for dataset_name, model_name in itertools.product(dataset_names, model_names):
        print(f'\n --- Dataset: {dataset_name} - Model: {model_name} ---\n')
        load_fn = dataset_name_to_load_fn[dataset_name]
        problem_class = dataset_name_to_problem_class[dataset_name]
        dataset_filepath = f'datasets/{dataset_name}.csv'
        hpt_fn = model_name_to_hpt_fn[model_name]

        data = load_fn(problem_class, dataset_filepath)
        results, best_model = hpt_fn(data)

        results.set_index('metric', inplace=True)
        with open(f'hpt/{dataset_name}_{model_name}.pkl', 'wb') as f:
            pickle.dump(results, f)

        with open(f'hpt_best_models/{dataset_name}_{model_name}.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        print('\n---\n')
