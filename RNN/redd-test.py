from __future__ import print_function, division

import time
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from nilmtk import DataSet, HDFDataStore
from rnndisaggregator import RNNDisaggregator


print("========== OPEN DATASETS ============")
# from nilmtk.dataset_converters import convert_redd
# convert_redd('../data/REDD/low_freq', '../data/REDD/redd.csv', format='CSV')
train = DataSet('../data/REDD/redd.csv', format='CSV')
train.clear_cache()
train.set_window(end="14-5-2011")
validation = DataSet('../data/REDD/redd.csv', format='CSV')
validation.clear_cache()
validation.set_window(start="14-5-2011", end="21-5-2011")
test = DataSet('../data/REDD/redd.csv', format='CSV')
test.clear_cache()
test.set_window(start="21-5-2011")

train_building = 1
validation_building = 1
test_building = 1
sample_period = 6
meter_key = 'fridge'

train_elec = train.buildings[train_building].elec
validation_elec = validation.buildings[validation_building].elec
test_elec = test.buildings[test_building].elec

train_meter = train_elec.submeters()[meter_key]
validation_meter = validation_elec.submeters()[meter_key]
test_meter = test_elec.submeters()[meter_key]

# meter 1 + 2
train_mains = train_elec.mains()
validation_mains = validation_elec.mains()
test_mains = test_elec.mains()

results_dir = '../results/REDD-RNN-{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(results_dir)
train_logfile = os.path.join(results_dir, 'training.log')
val_logfile = os.path.join(results_dir, 'validation.log')

rnn = RNNDisaggregator(train_logfile, val_logfile)

print("========== TRAIN ============")
epochs = 0
start = time.time()
for i in range(20):
    rnn.train(train_mains, train_meter, validation_mains, validation_meter, epochs=5, sample_period=sample_period)
    epochs += 5
    rnn.export_model(os.path.join(results_dir, "REDD-RNN-h{}-{}-{}epochs.h5".format(train_building, meter_key, epochs)))
    print("CHECKPOINT {}".format(epochs))
end = time.time()
print("Train =", end-start, "seconds.")

results_file = os.path.join(results_dir, 'results.txt')
headline = "========== DISAGGREGATE ============"
with open(results_file, "w") as text_file:
    text_file.write(headline + '\n')
print(headline)
disag_filename = 'disag-out.h5'
output = HDFDataStore(os.path.join(results_dir, disag_filename), 'w')
rnn.disaggregate(test_mains, output, results_file, train_meter, sample_period=sample_period)
output.close()

print("========== PLOTS ============")
result = DataSet(os.path.join(results_dir, disag_filename))
res_elec = result.buildings[test_building].elec

# plots
predicted = res_elec[meter_key]
ground_truth = test_elec[meter_key]
predicted.plot()
ground_truth.plot()
plt.savefig(os.path.join(results_dir, 'predicted_vs_ground_truth.png'))
plt.close()

training = pd.read_csv(train_logfile)
epochs = np.array(training.as_matrix()[:,0], dtype='int')
loss = np.array(training.as_matrix()[:,1], dtype='float32')
plt.plot(epochs, loss, label='train')
validation = pd.read_csv(val_logfile)
epochs = np.array(validation.as_matrix()[:,0], dtype='int')
loss = np.array(validation.as_matrix()[:,1], dtype='float32')
plt.plot(epochs, loss, label='validation')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(os.path.join(results_dir, 'loss.png'))
plt.close()
