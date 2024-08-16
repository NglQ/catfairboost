import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from pandas.api.types import is_categorical_dtype
from pre_processing_module import categorical_to_numerical
import os


def get_columns_names(name, data):
    os.makedirs(f'{name}', exist_ok=True)
    cols = data.columns
    str_to_save = f'{name}: \n{cols}'
    print(str_to_save, file=open(f'{name}/columns_names.txt', 'w'))
    return cols


def get_columns_types(name, data):
    os.makedirs(f'{name}', exist_ok=True)
    types = data.dtypes
    str_to_save = f'{name}: \n{types}'
    print(str_to_save, file=open(f'{name}/columns_dtypes.txt', 'w'))
    return types


def get_data_description(name, data):
    os.makedirs(f'{name}', exist_ok=True)
    description = data.describe()
    str_to_save = f'{name}: \n{description}'
    print(str_to_save, file=open(f'{name}/columns_description.txt', 'w'))
    return description


def show_histograms(name, data):
    os.makedirs(f'{name}', exist_ok=True)
    for column in filter(lambda x: is_categorical_dtype(data[x]), data.columns):
        sns.histplot(data[column])
        plt.title(f'{name} - {column}')
        plt.tight_layout()
        plt.savefig(f'{name}/{column}_histogram.png')
        plt.show()


def show_violin_plots(name, data):
    os.makedirs(f'{name}', exist_ok=True)
    for column in filter(lambda x: not is_categorical_dtype(data[x]), data.columns):
        sns.violinplot(data[column])
        plt.title(f'{name} - {column}')
        plt.savefig(f'{name}/{column}_violin_plot.png')
        plt.show()


def show_correlation_matrix(name, data):
    data_copy = data.copy()
    data_copy = categorical_to_numerical(data_copy)
    os.makedirs(f'{name}', exist_ok=True)
    sns.heatmap(data_copy.corr(), annot=True)
    plt.title(f'{name} - Correlation')
    plt.savefig(f'{name}/correlation_matrix.png')
    plt.show()


def plot_dataframe(name, data, labels=None, vmin=-1.95, vmax=1.95, figsize=(10, 10), s=4):
    data_copy = data.copy()
    data_copy = categorical_to_numerical(data_copy)
    labels_copy = labels.copy().to_frame()
    labels_copy = categorical_to_numerical(labels_copy)

    data_copy = (data_copy - data_copy.mean()) / data_copy.std()
    os.makedirs(f'{name}', exist_ok=True)

    plt.figure(figsize=figsize)
    plt.imshow(data_copy.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels_copy is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data_copy.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels_copy.index, np.ones(len(labels_copy)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(labels_copy))
    plt.title(f'{name} - General Description')
    plt.savefig(f'{name}/general_description.png')
    plt.show()


def plot_pca(name, data, labels, n_components=2):
    data_copy = data.copy()
    data_copy = categorical_to_numerical(data_copy)
    labels_copy = labels.copy().to_frame()
    labels_copy = categorical_to_numerical(labels_copy)
    os.makedirs(f'{name}', exist_ok=True)
    pca = PCA(n_components=n_components)
    pca.fit(data_copy)
    data_pca = pca.transform(data_copy)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels_copy[labels_copy.columns[0]], cmap='tab10')
    plt.title(f'{name} - PCA')
    plt.savefig(f'{name}/pca.png')
    plt.show()


def compute_mutual_information(name, data, target, cols_to_select, regression=False):
    os.makedirs(f'{name}', exist_ok=True)
    sensitive_data = data[cols_to_select]
    target_copy = target.copy().to_frame()
    target_copy = categorical_to_numerical(target_copy)

    sensitive_dataset = sensitive_data.copy()
    sensitive_dataset[target_copy.columns[0]] = target_copy[target_copy.columns[0]]
    sensitive_dataset.dropna(inplace=True)
    sensitive_dataset = categorical_to_numerical(sensitive_dataset)

    if regression:
        result = mutual_info_regression(sensitive_dataset.drop(columns=[target_copy.columns[0]]), sensitive_dataset[target_copy.columns[0]])
    else:
        result = mutual_info_classif(sensitive_dataset.drop(columns=[target_copy.columns[0]]), sensitive_dataset[target_copy.columns[0]])
    dict_result = dict(zip(cols_to_select, result))
    str_to_save = f'{name}: \n'
    for k, v in dict_result.items():
        str_to_save += f'{k}:{v}\n'
    print(str_to_save, file=open(f'{name}/mutual_information.txt', 'w'))
    return dict_result


def not_a_number_percentage(name, data):
    os.makedirs(f'{name}', exist_ok=True)
    nulls = data.isnull().sum().sum()
    total_examples = data.shape[0]
    nulls_perc = nulls/total_examples * 100
    str_to_save = f'{name}: \n{nulls_perc}%'
    print(str_to_save, file=open(f'{name}/Nan_percentage.txt', 'w'))
    return


def analyze_data(name, data, target, regression_task, sensitive_feats):
    dataset = data.copy()
    dataset['target'] = target
    print(get_columns_names(name, dataset))
    print(get_columns_types(name, dataset))
    print(get_data_description(name, dataset))
    print(not_a_number_percentage(name, dataset))
    plot_dataframe(name, data, labels=target)
    show_histograms(name, dataset)
    show_violin_plots(name, dataset)
    show_correlation_matrix(name, dataset)
    plot_pca(name, dataset, target)
    print(compute_mutual_information(name, data, target, sensitive_feats, regression=regression_task))
