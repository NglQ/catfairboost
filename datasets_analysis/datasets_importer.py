from fairlearn.datasets import fetch_adult, fetch_boston, fetch_acs_income, fetch_bank_marketing
from fairlearn.datasets import fetch_diabetes_hospital, fetch_credit_card
import pandas as pd


def import_datasets():
    datasets_fairness = [
        fetch_adult(),
        fetch_boston(),  # might not be a good dataset for fairness analysis
        fetch_acs_income(),
        fetch_bank_marketing(),  # might not be a good dataset for fairness analysis
        fetch_diabetes_hospital(),
        fetch_credit_card()]

    sensitive_features = [['race', 'sex', 'native-country'],  # fnlwgt might be a sensitive feature
                          ['CRIM', 'LSTAT'],
                          ['SEX', 'RAC1P', 'POBP'],
                          ['V1'],  # sensitive features not easy to identify
                          ['race', 'gender'],
                          ['x2', 'x3']]  # x2: Gender,
                                         # x3:Education,
                                         # (see if it is a sensitive feature, in such case add to the others as well)

    tasks = [False, True, True, False, False, False]

    datasets_fairness = [{'name': d_set.details['name'],
                          'data': d_set.data, 'target': d_set.target,
                          'sen_feat': sen_feat,
                          'regression_task': task}
                         for d_set, sen_feat, task in zip(datasets_fairness, sensitive_features, tasks)]

    diabetes_custom_dset = pd.read_csv('../diabetes_dataset.csv')  # might not be suitable for our purposes
    diabetes_custom_element = {'name': 'diabetes_race', 'data': diabetes_custom_dset.drop('diabetes', axis=1),
                               'target': diabetes_custom_dset['diabetes'],
                               'sen_feat': ['gender', 'race:AfricanAmerican', 'race:Asian',
                                            'race:Caucasian', 'race:Hispanic', 'race:Other'],
                               'regression_task': False}

    datasets_fairness.append(diabetes_custom_element)

    return datasets_fairness
