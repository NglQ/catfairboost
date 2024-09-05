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
        params = {
            'num_iterations': trial.suggest_int('num_iterations', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 100),
            'num_leaves': trial.suggest_int('num_leaves', 4, 64),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1.0, log=True)
        }

        base_params.update(params)

        model = lgb.LGBMClassifier(verbose=0)
        model.set_params(**base_params)

        model.fit(X_train, y_train)

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
        params = {
            'num_iterations': trial.suggest_int('num_iterations', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 100),
            'num_leaves': trial.suggest_int('num_leaves', 4, 64),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1.0, log=True),
            'multiplier_learning_rate': trial.suggest_float('multiplier_learning_rate', 5000, 20000, log=True)
        }

        base_params.update(params)

        fairgbm_clf = FairGBMClassifier(
            constraint_type='FNR,FPR',
            **base_params,
        )

        fairgbm_clf.fit(X_train, y_train, constraint_group=X_train[data['sf_name']].to_list())

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
            'iterations': trial.suggest_int('iterations', 10, 2000),
            'random_seed': 42,
            'cat_features': data['cat_cols'],
            'verbose': False,
            'thread_count': -1,
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 1, 16),
            'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
            'bootstrap_type': trial.suggest_categorical(
                'bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']
            )
        }

        if param['bootstrap_type'] == 'Bayesian':
            param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
        elif param['bootstrap_type'] == 'Bernoulli':
            param['subsample'] = trial.suggest_float('subsample', 0.1, 1, log=True)

        model = cb.CatBoostClassifier(**param)

        pruning_callback = CatBoostPruningCallback(trial, "Accuracy")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
        )

        # evoke pruning manually.
        pruning_callback.check_pruned()

        predictions = model.predict(X_val)
        y_pred = np.rint(predictions)
        accuracy = accuracy_score(y_val, y_pred)
        eod = equalized_odds_difference(y_val, y_pred, sensitive_features=X_val[data['sf_name']])

        print('Accuracy: ', accuracy)
        print('Equalized odds diff: ', eod)

        return accuracy

    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(sampler=sampler, pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def hpt_catboost_2(data):
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
            'iterations': trial.suggest_int('iterations', 10, 2000),
            'random_seed': 42,
            'cat_features': data['cat_cols'],
            'verbose': False,
            'thread_count': -1,
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 1, 16),
            'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
            'bootstrap_type': trial.suggest_categorical(
                'bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']
            )
        }

        if param['bootstrap_type'] == 'Bayesian':
            param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
        elif param['bootstrap_type'] == 'Bernoulli':
            param['subsample'] = trial.suggest_float('subsample', 0.1, 1, log=True)

        model = cb.CatBoostClassifier(**param)
        model.fit(X_train, y_train, verbose=0)

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
    hpt_catboost_2(data)
