import pickle
import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio
from error_parity import RelaxedThresholdOptimizer
from folktables import ACSIncome, ACSEmployment, ACSIncomePovertyRatio, ACSMobility

from params import lightgbm_params, fairgbm_params
from load_data import load_diabetes_easy_cat, load_diabetes_easy_gbm, load_diabetes_hard_cat, load_diabetes_hard_gbm, \
    load_acs_problem_cat, load_acs_problem_gbm
from training import LightFairGBM

warnings.filterwarnings('ignore', category=FutureWarning)

# Set seed for consistent results with ExponentiatedGradient
np.random.seed(42)

# TODO: compare fairly the fairness models (pun intended)
#  - find best hps according to accuracy
#  - train fair versions of these base models (does not apply to FairGBM)
#  - pickle and load them in this module
#  - "unprocess" (i.e., apply `error-parity` library with tolerance=+inf) the fair models so that can can be compared
#     when...
#  - postprocessing is applied (using `error-parity` again) with a specified tolerance != +inf
#  - plot pareto frontier

data = load_acs_problem_gbm(ACSMobility, 'datasets/acsmobility.csv')

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

with open('models/lightfairgbm.pkl', 'rb') as f:
    model = pickle.load(f)

# Given any trained model that outputs real-valued scores
fair_clf = RelaxedThresholdOptimizer(
    predictor=lambda X: model._pmf_predict(X)[:, 1],
    constraint="equalized_odds",  # other constraints are available
    tolerance=0.05,               # fairness constraint tolerance
)

# Fit the fairness adjustment on some data
# This will find the optimal _fair classifier_
fair_clf.fit(X=X_train, y=y_train, group=s_train)

# Now you can use `fair_clf` as any other classifier
# You have to provide group information to compute fair predictions
y_pred = fair_clf(X=X_test, group=s_test)

accuracy = accuracy_score(y_test, y_pred)
eod = equalized_odds_difference(y_test, y_pred, sensitive_features=X_test[data['sf_name']])

print('Accuracy: ', accuracy)
print('Equalized odds diff: ', eod)
