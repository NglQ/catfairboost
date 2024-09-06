import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import Pool

from folktables import ACSDataSource, generate_categories


def get_splits(data):
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

    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    s_train.reset_index(drop=True, inplace=True)
    s_val.reset_index(drop=True, inplace=True)
    s_test.reset_index(drop=True, inplace=True)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        's_train': s_train,
        's_val': s_val,
        's_test': s_test
    }


def split_data(data, target_name, sensitive_feature_name):
    """
    :return: the dataset splits (train 80%, test 20%) in the expected format
    """
    data_nunique = data.nunique()

    data_points = len(data)
    test_size = int(0.20 * data_points)

    X_train, X_test = train_test_split(data, test_size=test_size, random_state=42, shuffle=True,
                                       stratify=data[sensitive_feature_name])

    y_train = X_train[target_name]
    y_test = X_test[target_name]

    X_train = X_train.drop(columns=[target_name]).reset_index(drop=True)
    X_test = X_test.drop(columns=[target_name]).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, y_train, X_test, y_test, data_nunique


def convert_to_cat_datasets(data, target_name, sensitive_feature_name, categorical_columns):
    X_train, y_train, X_test, y_test, data_nunique = \
        split_data(data, target_name, sensitive_feature_name)

    train_data = Pool(X_train, label=y_train, cat_features=categorical_columns)
    test_data = Pool(X_test, label=y_test, cat_features=categorical_columns)

    return {
        'raw_data': {'train': X_train, 'test': X_test},
        'nunique': data_nunique,
        'cat_cols': categorical_columns,
        'train': train_data,
        'test': test_data,
        'target_name': target_name,
        'sensitive_feature_name': sensitive_feature_name,
        'sf_name': sensitive_feature_name,
        'sf_train': X_train[sensitive_feature_name],
        'sf_test': X_test[sensitive_feature_name]
    }


def convert_to_gbm_datasets(data):
    X_train = data['raw_data']['train']
    X_test = data['raw_data']['test']
    y_train = data['train'].get_label()
    y_test = data['test'].get_label()
    data_nunique = data['nunique']
    categorical_columns = data['cat_cols']

    def convert_categorical_to_int(dfs: dict, categorical_columns):
        df = pd.concat([dfs['train'], dfs['test']], axis=0).reset_index(drop=True)

        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            le.fit(df[col])
            label_encoders[col] = le
        return label_encoders

    label_encoders = convert_categorical_to_int(data['raw_data'], categorical_columns)

    for cat in categorical_columns:
        assert len(label_encoders[cat].classes_) == data_nunique[cat]

        X_train[cat] = label_encoders[cat].transform(X_train[cat])
        X_test[cat] = label_encoders[cat].transform(X_test[cat])

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(X_train.columns),
                             categorical_feature=categorical_columns, free_raw_data=False).construct()
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=list(X_test.columns),
                            categorical_feature=categorical_columns, free_raw_data=False).construct()

    data['train'] = train_data
    data['test'] = test_data
    data['sf_train'] = data['raw_data']['train'][data['sf_name']]
    data['sf_test'] = data['raw_data']['test'][data['sf_name']]

    return data


def load_diabetes_easy_cat(problem_class: None, dataset_filepath):
    """
    Source: https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset/data
    """

    data = pd.read_csv(dataset_filepath)

    target_name = 'diabetes'
    sensitive_feature_name = 'race'
    categorical_columns = ['gender', 'location', 'hypertension', 'heart_disease', 'smoking_history', 'race']

    old_race_features = ['race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other']
    to_drop = old_race_features + ['year']
    data['race'] = data[old_race_features].idxmax(axis=1).apply(lambda x: x.replace('race:', ''))
    data.drop(columns=to_drop, inplace=True)

    return convert_to_cat_datasets(data, target_name, sensitive_feature_name, categorical_columns)


def load_diabetes_hard_cat(problem_class: None, dataset_filepath):
    """
    Source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
    """

    data = pd.read_csv(dataset_filepath)

    target_name = 'readmitted'
    sensitive_feature_name = 'race'
    categorical_columns = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
                           'admission_source_id', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                           'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                           'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                           'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
                           'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                           'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
                           'diabetesMed']

    # Convert target into binary
    data['readmitted'] = data['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)

    data['max_glu_serum'] = data['max_glu_serum'].fillna('?')
    data['A1Cresult'] = data['A1Cresult'].fillna('?')
    to_drop = ['encounter_id', 'patient_nbr', 'weight']
    data.drop(columns=to_drop, inplace=True)

    return convert_to_cat_datasets(data, target_name, sensitive_feature_name, categorical_columns)


def load_acs_problem_cat(problem_class, dataset_filepath):
    """
    Source: https://github.com/socialfoundations/folktables
    """

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

    if os.path.isfile(dataset_filepath):
        data = pd.read_csv(dataset_filepath)
        definition_df = data_source.get_definitions(download=True)
        categories = generate_categories(features=problem_class.features, definition_df=definition_df)
    else:
        acs_data = data_source.get_data(states=['CA', 'TX', 'WA'], download=True)
        definition_df = data_source.get_definitions(download=True)
        categories = generate_categories(features=problem_class.features, definition_df=definition_df)
        X, y, sf = problem_class.df_to_pandas(acs_data, categories=categories)
        data = pd.concat([X, y], axis=1)
        data.to_csv(dataset_filepath, index=False)

    target_name = problem_class.target
    sensitive_feature_name = problem_class.group
    categorical_columns = list(categories)

    for col in data.columns:
        if (data[col] == -1).any() == True:
            data[col] = data[col].apply(lambda x: np.nan if x == -1 else x)
            print(f'Found column {col} with -1 values! Replacing them...')

        if data[col].isna().any() == True:
            mode = data[col].mode()[0]
            data[col] = data[col].fillna(mode)
            print(f'Found column {col} with nan values! Replacing them...')

    cols_to_convert = list(set(data.columns) - set(categories))
    for col in cols_to_convert:
        data[col] = data[col].astype('int64')

    return convert_to_cat_datasets(data, target_name, sensitive_feature_name, categorical_columns)


def load_diabetes_easy_gbm(problem_class: None, dataset_filepath):
    data = load_diabetes_easy_cat(problem_class, dataset_filepath)
    return convert_to_gbm_datasets(data)


def load_diabetes_hard_gbm(problem_class: None, dataset_filepath):
    data = load_diabetes_hard_cat(problem_class, dataset_filepath)
    return convert_to_gbm_datasets(data)


def load_acs_problem_gbm(problem_class, dataset_filepath):
    data = load_acs_problem_cat(problem_class, dataset_filepath)
    return convert_to_gbm_datasets(data)
