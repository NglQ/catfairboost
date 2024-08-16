from datasets_importer import import_datasets
from data_analysis_toolkit import analyze_data
from pre_processing_module import object_to_categorical

if __name__ == '__main__':
    datasets = import_datasets()

    # for dataset in datasets:
    #     dataset['data'] = object_to_categorical(dataset['data'])
    #
    #     analyze_data(dataset['name'], dataset['data'], dataset['target'],
    #                  dataset['regression_task'], dataset['sen_feat'])

    analyze_data(datasets[0]['name'], datasets[0]['data'], datasets[0]['target'],
                 datasets[0]['regression_task'], datasets[0]['sen_feat'])


# TODO: histograms need to be rescaled to accommodate the column's names
#       txt description needs to show all the columns
#       correlation matrix needs show the lower triangular matrix only all cells should be annotated
#       PCA plot should contain a legend
#       sometimes categorical columns are detected as numerical
