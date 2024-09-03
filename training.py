from time import time
import warnings

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

from params import lightgbm_params, fairgbm_params
from load_data import load_diabetes_easy_cat, load_diabetes_easy_gbm, load_diabetes_hard_cat, load_diabetes_hard_gbm, \
    load_acs_income_cat, load_acs_income_gbm

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Found `num_iterations` in params. Will use it instead of argument')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*ChainedAssignmentError.*')

# Set the display format for floating point numbers to 4 decimal places
pd.options.display.float_format = '{:.4f}'.format

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set seed for consistent results with ExponentiatedGradient
np.random.seed(42)

# TODO:
#  - implementing load functions for folktables
#  - hyperparameter tuning (optuna)
#  - implement unconstrained models with `error-parity` library
#  - training tests with all models
#  - pareto frontier plots


def train_base_lightgbm(data):
    print('--- START LightGBM ---')

    start = time()
    lightgbm_model = lgb.train(lightgbm_params, data['train'])
    end = time()

    y_pred_proba = lightgbm_model.predict(data['raw_data']['test'])
    # Convert probabilities to class labels
    y_pred = (y_pred_proba > 0.5).astype(int)

    training_time = end - start
    print('--- END LightGBM ---')

    return data, y_pred, training_time


def train_fair_lightgbm(data):
    print('--- START LightFairGBM ---')

    class LightFairGBM(lgb.LGBMClassifier):
        def fit(self, *args, **kwargs):
            kwargs['categorical_feature'] = data['cat_cols']
            super().fit(*args, **kwargs)

    model = LightFairGBM(verbose=0)
    model.set_params(**lightgbm_params)

    estimator = ExponentiatedGradient(
        estimator=model,
        constraints=EqualizedOdds()
    )
    start = time()
    estimator.fit(data['raw_data']['train'], data['train'].get_label(), sensitive_features=data['sf_train'])
    end = time()

    y_pred = estimator.predict(data['raw_data']['test'])

    training_time = end - start
    print('--- END LightFairGBM ---')

    return data, y_pred, training_time


def train_base_catboost(data):
    print('--- START CatBoost ---')

    model = CatBoostClassifier(iterations=100, random_seed=42, verbose=False, thread_count=-1)

    start = time()
    model.fit(data['train'])
    end = time()
    y_pred = model.predict(data['test'])

    training_time = end - start
    print('--- END CatBoost ---')

    return data, y_pred, training_time


def train_fair_catboost(data):
    print('--- START CatFairBoost ---')

    model = CatBoostClassifier(iterations=100, random_seed=42, cat_features=data['cat_cols'], verbose=False,
                               thread_count=-1)

    estimator = ExponentiatedGradient(
        estimator=model,
        constraints=EqualizedOdds()
    )
    start = time()
    estimator.fit(data['raw_data']['train'], data['train'].get_label(), sensitive_features=data['sf_train'])
    end = time()

    y_pred = estimator.predict(data['raw_data']['test'])

    training_time = end - start
    print('--- END CatFairBoost ---')

    return data, y_pred, training_time


def train_base_fairgbm(data):
    print('--- START FairGBM ---')

    X_train = data['raw_data']['train']
    X_test = data['raw_data']['test']
    y_train = data['train'].get_label()

    # Instantiate
    fairgbm_clf = FairGBMClassifier(
        constraint_type='FNR,FPR',
        **fairgbm_params,
    )

    start = time()
    fairgbm_clf.fit(X_train, y_train, constraint_group=X_train[data['sf_name']].to_list())
    end = time()

    # Predict
    y_pred_proba = fairgbm_clf.predict_proba(X_test)[:, -1]
    # Convert probabilities to class labels
    y_pred = (y_pred_proba > 0.5).astype(int)

    training_time = end - start
    print('--- END FairGBM ---')

    return data, y_pred, training_time


if __name__ == '__main__':
    data = load_acs_income_gbm()
    data, y_pred, training_time = train_fair_catboost(data)

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
