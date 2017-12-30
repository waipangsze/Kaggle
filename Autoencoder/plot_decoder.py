from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# testing plt
a = np.random.rand(1, 10) # or (1, 10) or (1,4,4,8) 
print(a.shape)
# plt.imshow(a, cmap=plt.cm.gray)
# plt.show()

# load
model = load_model('decoder.h5')
y = np.zeros((4, 784))
for p1 in range(4):
    y[p1, :] = model.predict(a)
y = y.reshape((4, 28,28))
print(y.shape)

# ploting
plt.subplot(2,2,1)
plt.imshow(y[0, :, :], cmap=plt.cm.gray)
plt.subplot(2,2,2)
plt.imshow(y[1, :, :], cmap=plt.cm.gray)
plt.subplot(2,2,3)
plt.imshow(y[2, :, :], cmap=plt.cm.gray)
plt.subplot(2,2,4)
plt.imshow(y[3, :, :], cmap=plt.cm.gray)
plt.savefig('0_'+str(np.random.random())+'.png')
plt.close()
