from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fairlearn.datasets import fetch_diabetes_hospital

import lightgbm as lgb


def load_diabetes(return_nunique=False):
    """
    This load function automatically prepares the Diabetes 130 Hospitals Dataset (Fairlearn version) for the training
    processes of our algorithms.
    :return: the dataset splits (train 70%, validation 15%, test 15%) in the expected format
    """

    # TODO: remember to handle the categorical features in LightGBM-based models

    data = fetch_diabetes_hospital().frame
    data.drop(columns=['readmitted', 'readmit_binary'], inplace=True)
    data_nunique = data.nunique()

    data_points = len(data)
    test_size = int(0.15 * data_points)

    X_train, X_test = train_test_split(data, test_size=test_size, random_state=42, shuffle=True,
                                       stratify=data['race'])

    X_train, X_val = train_test_split(X_train, test_size=test_size, random_state=42, shuffle=True,
                                      stratify=X_train['race'])

    y_train = X_train['readmit_30_days']
    y_val = X_val['readmit_30_days']
    y_test = X_test['readmit_30_days']

    X_train = X_train.drop(columns=['readmit_30_days']).reset_index(drop=True)
    X_val = X_val.drop(columns=['readmit_30_days']).reset_index(drop=True)
    X_test = X_test.drop(columns=['readmit_30_days']).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if return_nunique:
        return X_train, y_train, X_val, y_val, X_test, y_test, data_nunique
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test


def load_diabetes_gbm():
    X_train, y_train, X_val, y_val, X_test, y_test, data_nunique = load_diabetes(return_nunique=True)

    def convert_categorical_to_int(df, categorical_columns):
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        return df, label_encoders

    categorical_columns = ['race', 'gender', 'age', 'discharge_disposition_id', 'admission_source_id',
                           'medical_specialty', 'primary_diagnosis', 'max_glu_serum', 'A1Cresult', 'insulin', 'change',
                           'diabetesMed', 'medicare', 'medicaid', 'had_emergency', 'had_inpatient_days',
                           'had_outpatient_days']
    X_train, label_encoders = convert_categorical_to_int(X_train, categorical_columns)

    for cat in categorical_columns:
        assert len(label_encoders[cat].classes_) == data_nunique[cat]

        X_val[cat] = label_encoders[cat].transform(X_val[cat])
        X_test[cat] = label_encoders[cat].transform(X_test[cat])

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(X_train.columns),
                             categorical_feature=categorical_columns)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=list(X_val.columns),
                           categorical_feature=categorical_columns, reference=train_data)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=list(X_test.columns),
                            categorical_feature=categorical_columns)

    return train_data, val_data, test_data
