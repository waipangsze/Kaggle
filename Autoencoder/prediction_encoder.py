import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.models import load_model

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60 000, 28x28), y shape (10 000, )
(x_train, _), (x_test, y_test) = mnist.load_data()

# data pre-processing
x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

encoder = load_model('encoder.h5')

# plotting
encoded_imgs = encoder.predict(x_test)
plt.subplot(1,2,1)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1])
plt.title('Original data')
plt.subplot(1,2,2)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.title('+ human made label')
plt.savefig('original.png')
plt.close()