Nomad2018 Predicting Transparent Conductors: Predict the key properties of novel transparent semiconductors

https://www.kaggle.com/c/nomad2018-predict-transparent-conductors#description


1. using ANN(2 hidden layers) + dropout(0.5), validation_split =0.1, epochs=100, batch_size=50

loss function? mse, mae, mape, msle (mean_squared_logarithmic_error),...

"mae": <score = 0.1319>

Epoch 100/100
2160/2160 - loss: 0.3644 - msle: 0.0407 - val_loss: 0.3509 - val_msle: 0.0345

"mse":

Epoch 77/100
2160/2160 - loss: 0.3035 - msle: 0.0360 - val_loss: 0.2240 - msle: 0.0267

"mae": Using np.max !!! < score = 0.1140>

Epoch 25/100
2160/2160 - loss: 0.1985 - msle: 0.0161 - val_loss: 0.1384 - val_msle: 0.0092

2) using Conv1D:

| Layer (type)              |   Output Shape  |            Param #   |
|-------|---------|---------|
|input_1 (InputLayer)     |    (None, 10, 1)      |       0   |      
|conv1d_1 (Conv1D)          |  (None, 7, 128)        |    640       |
|max_pooling1d_1 (MaxPooling1D) |(None, 3, 128)      |      0     |    
|conv1d_2 (Conv1D)       |     (None, 1, 128)     |       49280    | 
|global_average_pooling1d) |( (None, 128)       |        0     |    
|dense_1 (Dense)        |      (None, 2)        |         258    |   

|Total params: 50,178|
|---------|
|Trainable params: 50,178|
|Non-trainable params: 0|

"mae": Using np.max < score = 0.0780>
Epoch 84/100
2160/2160- loss: 0.0982 - msle: 0.0069 - val_loss: 0.0893 - val_msle: 0.0049
