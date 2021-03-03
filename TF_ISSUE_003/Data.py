import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from tensorflow import keras

def get_filepaths(project_path):
    results_path = project_path + '\\Results'
    charts_path = project_path + '\\Charts'
    dataset_path = project_path + '\\Data'
    predictions_path = project_path + '\\Predictions'
    filepaths_dict = {
        "Results": results_path,
        "Charts": charts_path,
        "Datasets": dataset_path,
        "Predictions": predictions_path
    }
    return filepaths_dict

def import_data(filepaths_dict):
    all_files = glob.glob(filepaths_dict.get('Datasets') + '/*.csv')
    files_list = []

    for this_file in all_files:
        df = pd.read_csv(this_file)
        files_list.append(df)
    #price_data = pd.concat(files_list, axis=0, ignore_index=True)
    return files_list

def split_data(program_params, train_data, train_labels):
    test_size = program_params.get('Test Samples') / train_data.shape[0]
    shuffle = program_params.get('Shuffle')
    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=test_size, shuffle=shuffle, stratify=None)
    '''
    train_data = train_data.values
    test_data = test_data.values
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
   '''
    return train_data, test_data, train_labels, test_labels

def save_data(filepaths, candle_types_df, model, program_params, model_params):
    #keras.utils.plot_model(model, to_file=filepaths.get('Results') + program_params.get('Model Name') + '-' + str(model_params['Experiment Num']), show_shapes=True, show_layer_names=True, rankdir='TB', dpi=96)
    candle_types_df.to_csv(filepaths.get('Results') + '\\' + program_params.get('Model Name') + '-' + str(model_params['Experiment #']) + '.csv')
    return