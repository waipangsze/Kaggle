import numpy as np
import keras, tensorflow
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input

def load_ann(m):
    input_data = Input(shape=(m,))
    x = Dense(128, activation='relu')(input_data)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_data = Dense(2, activation='relu')(x)
    
    # construct model
    model = Model(input=input_data, output=output_data)
    
    return model

def load_cnn1(m):
    input_data = Input(shape=(m, 1))
    x = Conv1D(128, 4, activation='relu')(input_data)
    x = MaxPooling1D()(x) # keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    output_data = Dense(2, activation='relu')(x)
    # construct model
    model = Model(input=input_data, output=output_data)
    
    return model