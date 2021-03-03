import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, add, Flatten, Reshape, LSTM, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
#from tcn import TCN

import json
import os
import traceback

def build_residual_block(input1, dilation, filters, drop_rate):
    tanh_conv1d = Conv1D(filters, 1, dilation_rate=dilation, padding='causal', activation=tf.nn.tanh)(input1)
    tahn_drop = Dropout(drop_rate)(tanh_conv1d)
    sigmoid_conv1d = Conv1D(filters, 1, dilation_rate=dilation, padding='causal', activation=tf.nn.sigmoid)(input1)
    sig_drop = Dropout(drop_rate)(sigmoid_conv1d)
    mult = layers.Multiply()([tahn_drop,sig_drop])
    skip = Conv1D(1,1)(mult)
    residual = add([input1, skip])
    return residual, skip

def Model_1(model_params):
    input_shape = int((model_params['Lookback']+1))
    n_features = int(model_params['Num Cryptos']+1)
    embed_dim = int(model_params['Embed Dim'])
    input_len = int(model_params['Lookback']+1)
    n_layers = int(model_params['Num Layers'])
    drop_rate = model_params['Drop Rate']
    filters = int(model_params['Num Filters'])

    model_in = Input(shape=(input_shape,))
    embed_layer = keras.layers.Embedding(n_features, embed_dim, input_length=input_len)(model_in)   
    input_conv = Conv1D(filters, 1, padding='valid')(embed_layer)
    skip_list = []
    dilations = 1
    block, skip = build_residual_block(input_conv, dilations, filters, drop_rate)
    skip_list.append(skip)
    for x in range(n_layers):
        dilations = dilations * 2
        block, skip = build_residual_block(block, dilations, filters, drop_rate)
        skip_list.append(skip)

    final_add = add(skip_list)
    final1 = Conv1D(1, 1, activation=tf.nn.relu)(final_add)
    final2 = Conv1D(1, 1, activation=tf.nn.relu)(final1)
    flat = Flatten()(final2)
    model_out = Dense(1, activation=tf.nn.sigmoid)(flat)
    model = Model(model_in, model_out)
    return model

def set_model_num(model_params):

    this_model = {
        1:Model_1(model_params)
    }

    model = this_model.get(model_params['Model Type'])

    return model