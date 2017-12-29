import pandas as pd
from pandas import Series, DataFrame
import numpy as np

def load_data():
    #=======================================
    # training data
    #=======================================
    print('='*40)
    df = pd.read_csv('train.csv', header=0)
    df.info()
    # remove the useless data
    df = df.drop(['id', 'spacegroup'], axis = 1)
    df.info()

    train_data = df.values
    m = 10
    x_train, y_train = train_data[:, :m], train_data[:, m:]
    # like to be Normalized, or (np.sqrt(np.sum((...) ** 2)))
    # max is better
    for p1 in range(m):
        x_train[:, p1] = x_train[:, p1]/(np.max(x_train[:, p1]))
    print("X , Y train = ", x_train.shape, y_train.shape)
    print(x_train[:2, :])
    print(y_train[:10, :])
    
    #=======================================
    # testing data
    #=======================================
    print('='*40)
    df = pd.read_csv('test.csv', header=0)
    df = df.drop(['id', 'spacegroup'], axis = 1)
    df.info()
    test = df.values
    for p1 in range(m):
        test[:, p1] = test[:, p1]/(np.max(test[:, p1]))
    print("test = ", test.shape)
    print('='*10, 'loading data', '='*30)
    
    return x_train, y_train, test, m

