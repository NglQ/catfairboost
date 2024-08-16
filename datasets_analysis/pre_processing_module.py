from pandas.api.types import is_categorical_dtype


def categorical_to_numerical(data):
    for column in data.columns:
        if is_categorical_dtype(data[column]) or data[column].dtype == 'object':
            data[column] = data[column].astype('category').cat.codes

    return data


def object_to_categorical(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].astype('category')

    return data