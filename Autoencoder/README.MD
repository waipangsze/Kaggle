Autoencoder is an unsupervised learning which doesn't contain corresponding label and consists of encoder and decoder.\
A MNIST still is applied.\
autoencoder.py: a fully connected network and save autoencoder and encoder model, original.png is shown the enocder results.(here we use 2 dimension of encode layer for visible ploting.)\
decoder.py: a fully connected network and save decoder model, training for a particular input like digit 4. The decoder model generates some examples automatically. (decoder_examples.png)


Dataset: http://cis.jhu.edu/~sachin/digit/digit.html

Using:

- physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:04:00.0, compute capability: 6.1)\
Epoch 300/300 \
900/900 - 0s 42us/step - loss: 0.0236 - val_loss: 0.0302\
time =  14.532356262207031  s

- Using 5000% CPU, Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz \
Epoch 300/300 \
900/900 - 2s 2ms/step - loss: 0.0172 - val_loss: 0.0238 \
time =  568.6107196807861  s
