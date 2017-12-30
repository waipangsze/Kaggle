import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from keras.models import load_model
import pre_processing

test, m = pre_processing.load_test_data()
model = load_model('weights-improvement-25-0.1985.hdf5')

results = model.predict(test)
print("results = ", results.shape)

# test[:, 0] = id
data = {"formation_energy_ev_natom":results[:,0], "bandgap_energy_ev":results[:,1]}
df = DataFrame(data, columns =['formation_energy_ev_natom', 'bandgap_energy_ev'])
df.info()
df.index += 1 
df.to_csv('submit.csv', sep=',', encoding='utf-8', index_label='id')
