import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import Pool


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

    return data


def load_diabetes_easy_cat():
    data = pd.read_csv('datasets/diabetes_easy.csv')

    target_name = 'diabetes'
    sensitive_feature_name = 'race'
    categorical_columns = ['gender', 'location', 'hypertension', 'heart_disease', 'smoking_history', 'race']

    old_race_features = ['race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other']
    to_drop = old_race_features + ['year']
    data['race'] = data[old_race_features].idxmax(axis=1).apply(lambda x: x.replace('race:', ''))
    data.drop(columns=to_drop, inplace=True)

    return convert_to_cat_datasets(data, target_name, sensitive_feature_name, categorical_columns)


def load_diabetes_hard_cat():
    """
    Source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
    """
    data = pd.read_csv('datasets/diabetes_hard.csv')

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


def load_diabetes_easy_gbm():
    data = load_diabetes_easy_cat()
    return convert_to_gbm_datasets(data)


def load_diabetes_hard_gbm():
    data = load_diabetes_hard_cat()
    return convert_to_gbm_datasets(data)
