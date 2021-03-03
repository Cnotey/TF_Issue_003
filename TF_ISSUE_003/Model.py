import tensorflow as tf
from tensorflow import keras
import Model_Architecture
import numpy as np
import pandas as pd

def build_model(program_params, filepaths, model_params):
    model = Model_Architecture.set_model_num(model_params)
    optimizer = keras.optimizers.Adam(lr=model_params['Learning Rate'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    return model

def train_model(model_params, train_data, train_labels, test_data, test_labels, model, program_params):
    epochs = int(model_params['Num Epochs'])
    batch_size = int(model_params['Batch Size'])
    validation_split = program_params.get('Validation Samples') / train_data.shape[0]
    #es = keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=10), callbacks=[es]
    #model.summary()
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    return history, test_loss, test_accuracy

def predict(test_data, model):
    predictions = model.predict(test_data)
    return predictions

def get_model_architecture(model):
    model_architecture = pd.DataFrame(columns=['Layer', 'Units', 'Activation'])
    model_dict = model.get_config()
    for x in model_dict.get('layers'):
        class_name = x.get('class_name')
        config = x.get('config')
        units = config.get('units')
        activation = config.get('activation')
        model_architecture = model_architecture.append({'Layer':class_name, 'Units':units, 'Activation':activation}, ignore_index=True)
    return model_architecture