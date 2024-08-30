from time import time
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count, selection_rate, false_positive_rate, false_negative_rate, equalized_odds_difference
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt

from load_data import load_diabetes_easy_cat, load_diabetes_easy_gbm, load_diabetes_hard_cat, load_diabetes_hard_gbm

# Set the display format for floating point numbers to 4 decimal places
pd.options.display.float_format = '{:.4f}'.format

# TODO:
#  - training test with FairGBM
#  - implementing load functions for folktables
#  - training tests with all models

if __name__ == '__main__':
    # --- START LightGBM ---

    # data = load_diabetes_hard_gbm()
    #
    # lightgbm_params = {
    #     'num_iterations': 100,
    #     'objective': 'binary',
    #     'device_type': 'cpu',
    #     'num_threads': 8,
    #     'seed': 42,
    #     'deterministic': 'true'
    # }
    #
    # start = time()
    # lightgbm_model = lgb.train(lightgbm_params, data['train'], valid_sets=[data['val']])
    # end = time()
    #
    # y_pred_proba = lightgbm_model.predict(data['raw_data']['test'])
    # # Convert probabilities to class labels
    # y_pred = (y_pred_proba > 0.5).astype(int)

    # --- END LightGBM ---

    # --- START CatBoost ---

    data = load_diabetes_hard_cat()

    model = CatBoostClassifier(iterations=100, random_seed=42, verbose=False)

    start = time()
    model.fit(data['train'], eval_set=data['val'])
    end = time()
    y_pred = model.predict(data['test'])

    # --- END CatBoost ---

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
    print('Training time: ', end - start)
