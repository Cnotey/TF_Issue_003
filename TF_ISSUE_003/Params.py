import numpy as np
import pandas as pd
import random
import math

program_params = {
    'Model Name': 'PAS_v2',
    'Initial Population': 200,
    'Test Samples': 90,
    'Short Test Samples': 15,
    'Validation Samples': 1,
    'Shuffle': False,
    'Not Improved Limit': 10,
}

def gen_model_props(exp_num, generation):
    data = [exp_num, generation, 0, 0, 0, False]
    index = ['Experiment #', 'Generation', 'Train Accuracy', 'Test Accuracy', 'Num Cryptos', 'Fit?']
    model_properties = pd.Series(data=data, index=index)
    return model_properties

def gen_rand_params(exp_num):
    lrs = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    filters_list = [1, 2, 4, 8, 16, 32, 64]
    learning_rate = lrs[random.randrange(0, len(lrs), 1)]
    model_type = 1
    n_buckets = random.randrange(5, 100, 5)
    lookback = random.randrange(1, 100, 5)
    batch_size = random.randrange(100, 2500, 100)
    n_epochs = random.randrange(10, 400, 10)
    embed_dim = random.randrange(8, 1024, 16)
    drop_rate = (random.randrange(1, 6, 1)) * .1
    n_layers = random.randrange(3, 12, 1)
    n_filters = filters_list[random.randrange(0, len(filters_list), 1)]

    data = [exp_num, 1, 0, 0, 0, 0, 0, 0, 0, True, learning_rate, model_type, n_buckets, lookback, batch_size, n_epochs, embed_dim, drop_rate, n_layers, n_filters]
    column_names = [
        'Experiment #',
        'Generation',
        'Train Accuracy',
        'Test Accuracy',
        'Short Test Accuracy',
        'Rolling Window Accuracy',
        'Val Accuracy',
        'Fitness Value',
        'Num Cryptos',
        'Fit?',
        'Learning Rate', 
        'Model Type', 
        'Num Buckets', 
        'Lookback', 
        'Batch Size', 
        'Num Epochs', 
        'Embed Dim', 
        'Drop Rate', 
        'Num Layers',
        'Num Filters']

    model_params = pd.Series(data=data, index=column_names)

    return model_params