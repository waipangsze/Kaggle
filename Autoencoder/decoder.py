import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.utils import plot_model
import matplotlib.pyplot as plt

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60 000, 28x28), y shape (10 000, )
# (x_train, _), (x_test, y_test) = mnist.load_data()
x_train = np.loadtxt(open("data4.csv"), delimiter=',')

# data pre-processing
x_train = x_train.astype('float32') / 255.       # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
print(x_train.shape)

# in order to plot in a 2D figure
# now, make it as 10
encoding_dim = 10

# this is our input placeholder
input_img = Input(shape=(784,))

# encoder layers
encoded = Dense(228, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(228, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)

# construct the decoder model
# encoded_input must be form of a Keras Input layer
# retrieve the last layer of the autoencoder model
encoded_input = Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-4]
decoder_layer2 = autoencoder.layers[-3]
decoder_layer3 = autoencoder.layers[-2]
decoder_layer4 = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input)))))

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training, fit(X, Y), X and Y are same vector
autoencoder.fit(x_train, x_train, nb_epoch=300, batch_size=200, validation_split=0.1, shuffle=True)

# autoencoder.save('autoencoder.h5')
# encoder.save('encoder.h5')
decoder.save('decoder.h5')
# plot network
# plot_model(model, to_file='model.png')