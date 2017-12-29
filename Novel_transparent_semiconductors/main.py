import numpy as np
import keras, tensorflow
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint
import pre_processing
from pandas import Series, DataFrame

x_train, y_train, test, m = pre_processing.load_data()

# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
# Dense(): fully connected layer.
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=m))
model.add(Dropout(0.5))
model.add(Dense(164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='relu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# sgd, adam
model.compile(loss='mae', optimizer=sgd, metrics=['msle'])

# define the checkpoint
filepath="./weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train the model, iterating on the data in batches of 128 samples
history = model.fit(x_train, y_train, validation_split =0.1, epochs=100, batch_size=50, callbacks=callbacks_list)
# plot metrics
'''
pyplot.plot(history.history['mean_squared_error'])
pyplot.plot(history.history['mean_absolute_error'])
pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.show()
'''
