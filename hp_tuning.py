import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import equalized_odds_difference
import lightgbm as lgb
import catboost as cb
from fairgbm import FairGBMClassifier
import optuna
from optuna.integration import CatBoostPruningCallback
from folktables import ACSIncome, ACSEmployment, ACSIncomePovertyRatio, ACSMobility

from params import lightgbm_params, fairgbm_params
from load_data import load_diabetes_easy_cat, load_diabetes_easy_gbm, load_diabetes_hard_cat, load_diabetes_hard_gbm, \
    load_acs_problem_cat, load_acs_problem_gbm

np.random.seed(42)


def hpt_lightgbm(data):
    X_other = data['raw_data']['train']
    X_test = data['raw_data']['test']
    y_other = data['train'].get_label()
    y_test = data['test'].get_label()
    s_other = data['sf_train']
    s_test = data['sf_test']

    data_points = int(0.5 * len(X_test))

    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(X_other, y_other, s_other,
                                                                      test_size=data_points, random_state=42,
                                                                      shuffle=True, stratify=s_other)
    base_params = lightgbm_params.copy()

    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 5, 20)

        params = {
            'num_iterations': trial.suggest_int('num_iterations', 20, 300),
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', 5, 2**max_depth),
            'max_bin': trial.suggest_int('max_bin', 200, 350),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'boosting': trial.suggest_categorical('boosting', ['gbdt', 'dart']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 40),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 0.1, log=True),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.0, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 30),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 0.1, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 0.1, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-5, 0.1, log=True),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'path_smooth': trial.suggest_float('path_smooth', 0.0, 0.1, log=True)
        }

        base_params.update(params)

        model = lgb.LGBMClassifier(verbose=0)
        model.set_params(**base_params)

        model.fit(X_train, y_train, categorical_feature=data['cat_cols'])

        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        eod = equalized_odds_difference(y_val, y_pred, sensitive_features=X_val[data['sf_name']])

        print('Accuracy: ', accuracy)
        print('Equalized odds diff: ', eod)

        return accuracy

    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)


def hpt_fairgbm(data):
    X_other = data['raw_data']['train']
    X_test = data['raw_data']['test']
    y_other = data['train'].get_label()
    y_test = data['test'].get_label()
    s_other = data['sf_train']
    s_test = data['sf_test']

    data_points = int(0.5 * len(X_test))

    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(X_other, y_other, s_other,
                                                                      test_size=data_points, random_state=42,
                                                                      shuffle=True, stratify=s_other)
    base_params = fairgbm_params.copy()

    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 5, 20)

        params = {
            'num_iterations': trial.suggest_int('num_iterations', 20, 300),
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', 5, 2 ** max_depth),
            'max_bin': trial.suggest_int('max_bin', 200, 350),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'boosting': trial.suggest_categorical('boosting', ['gbdt', 'dart']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 40),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 0.1, log=True),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.0, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 30),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 0.1, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 0.1, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-5, 0.1, log=True),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'path_smooth': trial.suggest_float('path_smooth', 0.0, 0.1, log=True),
            'multiplier_learning_rate': trial.suggest_float('multiplier_learning_rate', 1e-5, 1000, log=True)
        }

        base_params.update(params)

        fairgbm_clf = FairGBMClassifier(
            constraint_type='FNR,FPR',
            **base_params,
        )

        fairgbm_clf.fit(X_train, y_train, constraint_group=X_train[data['sf_name']].to_list(),
                        categorical_feature=data['cat_cols'])

        y_pred_proba = fairgbm_clf.predict_proba(X_val)[:, -1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        eod = equalized_odds_difference(y_val, y_pred, sensitive_features=X_val[data['sf_name']])

        print('Accuracy: ', accuracy)
        print('Equalized odds diff: ', eod)

        return max(0, accuracy - eod)

    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)


def hpt_catboost(data):
    X_other = data['raw_data']['train']
    X_test = data['raw_data']['test']
    y_other = data['train'].get_label()
    y_test = data['test'].get_label()
    s_other = data['sf_train']
    s_test = data['sf_test']

    data_points = int(0.5 * len(X_test))

    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(X_other, y_other, s_other,
                                                                      test_size=data_points, random_state=42,
                                                                      shuffle=True, stratify=s_other)

    def objective(trial):
        param = {
            'eval_metric': 'Accuracy',
            'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
            'iterations': trial.suggest_int('iterations', 10, 300),
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
        model.fit(X_train, y_train, cat_features=data['cat_cols'], verbose=0)

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        eod = equalized_odds_difference(y_val, y_pred, sensitive_features=X_val[data['sf_name']])

        print('Accuracy: ', accuracy)
        print('Equalized odds diff: ', eod)

        return accuracy

    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)


if __name__ == '__main__':
    data = load_acs_problem_gbm(ACSMobility, 'datasets/acsmobility.csv')
    hpt_catboost(data)
