import numpy as np
import keras, tensorflow
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from pandas import Series, DataFrame
from keras.models import Model
from keras.layers import Input
import pre_processing
import load_model

x_train, y_train, m = pre_processing.load_train_data()

# if use cnn1 ========
x_train = np.expand_dims(x_train, axis=2)
print(x_train.shape)
# ====================

model = load_model.load_cnn1(m)

# model = load_model.load_ann(m)

# optimization
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

model.compile(loss='mae', optimizer=sgd, metrics=['msle'])

# define the checkpoint
filepath="./weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train the model, iterating on the data in batches of 128 samples
history = model.fit(x_train, y_train, validation_split =0.1, epochs=100, batch_size=50, shuffle=True, callbacks=callbacks_list)
# plot metrics
'''
pyplot.plot(history.history['mean_squared_error'])
pyplot.plot(history.history['mean_absolute_error'])
pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.show()
'''

print(model.summary())
