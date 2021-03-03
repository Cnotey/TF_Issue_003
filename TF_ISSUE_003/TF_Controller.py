import pandas as pd
import numpy as np
import Data
import Model

def train_model(program_params, model_params, filepaths_dict, data, labels):
    train_data, test_data, train_labels, test_labels = Data.split_data(program_params, data, labels)
    model = Model.build_model(program_params, filepaths_dict, model_params)
    history, test_loss, test_accuracy = Model.train_model(model_params, train_data, train_labels, test_data, test_labels, model, program_params)
    predictions = Model.predict(test_data, model)
    return model_params, predictions