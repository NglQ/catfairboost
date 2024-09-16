from time import time
import warnings
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from fairgbm import FairGBMClassifier
from catboost import CatBoostClassifier
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count, selection_rate, false_positive_rate, false_negative_rate, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
from folktables import ACSIncome, ACSEmployment, ACSIncomePovertyRatio, ACSMobility

from load_data import load_diabetes_easy_cat, load_diabetes_easy_gbm, load_diabetes_hard_cat, load_diabetes_hard_gbm, \
    load_acs_problem_cat, load_acs_problem_gbm, get_splits

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Found `num_iterations` in params. Will use it instead of argument')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*ChainedAssignmentError.*')
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')

# Set the display format for floating point numbers to 4 decimal places
pd.options.display.float_format = '{:.4f}'.format

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set seed for consistent results with ExponentiatedGradient
np.random.seed(42)


def train_base_lightgbm(data, data_splits, training_params):
    print('--- START LightGBM ---')

    model = lgb.LGBMClassifier(verbosity=-1)
    model.set_params(**training_params)

    start = time()
    model.fit(data_splits['X_train'], data_splits['y_train'], categorical_feature=data['cat_cols'])
    end = time()

    y_pred = model.predict(data_splits['X_test'])

    training_time = end - start
    print('--- END LightGBM ---')

    return model, data, y_pred, training_time


class LightFairGBM(lgb.LGBMClassifier):
    cat_cols = None

    def fit(self, *args, **kwargs):
        kwargs['categorical_feature'] = self.cat_cols
        super().fit(*args, **kwargs)


def train_fair_lightgbm(data, data_splits, training_params):
    print('--- START LightFairGBM ---')

    model = LightFairGBM(verbosity=-1)
    training_params.update({'cat_cols': data['cat_cols']})
    model.set_params(**training_params)

    estimator = ExponentiatedGradient(
        estimator=model,
        constraints=EqualizedOdds()
    )
    start = time()
    estimator.fit(data_splits['X_train'], data_splits['y_train'], sensitive_features=data_splits['s_train'])
    end = time()

    y_pred = estimator.predict(data_splits['X_test'])

    training_time = end - start
    print('--- END LightFairGBM ---')

    return estimator, data, y_pred, training_time


def train_base_catboost(data, data_splits, training_params):
    print('--- START CatBoost ---')

    training_params.update({'cat_features': data['cat_cols']})

    # In `training_params` kwargs dict expecting value `cat_features`
    assert 'cat_features' in training_params, 'Expected `cat_features` in training_params'
    model = CatBoostClassifier(**training_params)

    start = time()
    model.fit(data_splits['X_train'], data_splits['y_train'])
    end = time()
    y_pred = model.predict(data_splits['X_test'])

    training_time = end - start
    print('--- END CatBoost ---')

    return model, data, y_pred, training_time


def train_fair_catboost(data, data_splits, training_params):
    print('--- START CatFairBoost ---')

    training_params.update({'cat_features': data['cat_cols']})

    # In `training_params` kwargs dict expecting value `cat_features`
    assert 'cat_features' in training_params, 'Expected `cat_features` in training_params'
    model = CatBoostClassifier(**training_params)

    estimator = ExponentiatedGradient(
        estimator=model,
        constraints=EqualizedOdds()
    )
    start = time()
    estimator.fit(data_splits['X_train'], data_splits['y_train'], sensitive_features=data_splits['s_train'])
    end = time()

    y_pred = estimator.predict(data_splits['X_test'])

    training_time = end - start
    print('--- END CatFairBoost ---')

    return estimator, data, y_pred, training_time


def train_base_fairgbm(data, data_splits, training_params):
    print('--- START FairGBM ---')

    # Instantiate
    fairgbm_clf = FairGBMClassifier(
        constraint_type='FNR,FPR',
        **training_params,
    )

    start = time()
    fairgbm_clf.fit(data_splits['X_train'], data_splits['y_train'], constraint_group=data_splits['s_train'].to_list(),
                    categorical_feature=data['cat_cols'])
    end = time()

    # Predict
    y_pred_proba = fairgbm_clf.predict_proba(data_splits['X_test'])[:, -1]
    # Convert probabilities to class labels
    y_pred = (y_pred_proba > 0.5).astype(int)

    training_time = end - start
    print('--- END FairGBM ---')

    return fairgbm_clf, data, y_pred, training_time


if __name__ == '__main__':
    lightgbm_params = {
        'num_iterations': 100,
        'objective': 'binary',
        'device_type': 'cpu',
        'num_threads': 8,
        'seed': 42,
        'deterministic': 'true'
    }

    fairgbm_params = {
        'multiplier_learning_rate': 0.005,
        'num_iterations': 100,
        'device_type': 'cpu',
        'num_threads': 8,
        'seed': 42,
        'deterministic': 'true'
    }

    catboost_params = {
        'iterations': 100,
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1
    }

    data = load_diabetes_easy_gbm(None, 'datasets/diabetes_easy.csv')
    data_splits = get_splits(data)
    model, data, y_pred, training_time = train_base_lightgbm(data, data_splits, lightgbm_params)

    y_true = data['test'].get_label()
    sensitive_features = data['sf_test']

    metrics = {
        'count': count,
        'sel': selection_rate,
        'pre': precision_score,
        'acc': accuracy_score,
        'rec/tpr': recall_score,
        'fnr': false_negative_rate,
        'fpr': false_positive_rate
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    mf.by_group.plot.bar(
        subplots=True,
        figsize=[8, 18]
    )
    plt.show()

    print(mf.overall)
    print(mf.by_group)

    print('Equalized odds difference (the lower the better): ',
          equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features))
    print('F1 score: ', f1_score(y_true, y_pred))
    print('Training time: ', training_time)
