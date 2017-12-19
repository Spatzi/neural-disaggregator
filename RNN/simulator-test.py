from __future__ import print_function, division
import time
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from nilmtk import DataSet, HDFDataStore
import metrics
from rnndisaggregator import RNNDisaggregator


print("========== OPEN DATASETS ============")
# from nilmtk.dataset_converters import convert_redd
# convert_simulator('../data/SIMULATOR/low_freq', '../data/SIMULATOR/simulator.csv', format='CSV')
train = DataSet('../data/SIMULATOR/simulator.csv', format='CSV')
train.clear_cache()
train.set_window(end="24-4-2011")
validation = DataSet('../data/SIMULATOR/simulator.csv', format='CSV')
validation.clear_cache()
validation.set_window(start="24-4-2011", end="25-4-2011 12:00:00")
test = DataSet('../data/SIMULATOR/simulator.csv', format='CSV')
test.clear_cache()
test.set_window(start="25-4-2011 12:00:00")

train_building = 1
validation_building = 1
test_building = 1
sample_period = 1
meter_key = 'dish washer'
train_elec = train.buildings[train_building].elec
validation_elec = validation.buildings[validation_building].elec
test_elec = test.buildings[test_building].elec

train_meter = train_elec.submeters()[meter_key]
validation_meter = validation_elec.submeters()[meter_key]
test_meter = test_elec.submeters()[meter_key]

train_mains_instance = train_meter.upstream_meter().instance()
validation_mains_instance = validation_meter.upstream_meter().instance()
test_mains_instance = test_meter.upstream_meter().instance()

train_mains = train_elec.mains().all_meters()[train_mains_instance-1]
validation_mains = validation_elec.mains().all_meters()[validation_mains_instance-1]
test_mains = test_elec.mains().all_meters()[test_mains_instance-1]

results_dir = '../results/SIMULATOR-RNN-{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(results_dir)
train_logfile = os.path.join(results_dir, 'training.log')
val_logfile = os.path.join(results_dir, 'validation.log')

rnn = RNNDisaggregator(train_logfile, val_logfile)

start = time.time()
print("========== TRAIN ============")
epochs = 0
for i in range(2):
    rnn.train(train_mains, train_meter, validation_mains, validation_meter, epochs=2, sample_period=sample_period)
    epochs += 2
    rnn.export_model(os.path.join(results_dir, "SIMULATOR-RNN-h{}-{}-{}epochs.h5".format(train_building, meter_key, epochs)))
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

headline = "========== RESULTS ============"
with open(results_file, "a") as text_file:
    text_file.write(headline + '\n')
print(headline)
result = DataSet(os.path.join(results_dir, disag_filename))
res_elec = result.buildings[test_building].elec
rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])

lines = ["============ Recall: {}".format(rpaf[0]),
         "============ Precision: {}".format(rpaf[1]),
         "============ Accuracy: {}".format(rpaf[2]),
         "============ F1 Score: {}".format(rpaf[2]),
         "============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec[meter_key], test_elec[meter_key])),
         "============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec[meter_key], test_elec[meter_key]))
         ]

with open(results_file, "a") as text_file:
    for line in lines:
        text_file.write("%s\n" % line)

for line in lines:
    print(line)

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
