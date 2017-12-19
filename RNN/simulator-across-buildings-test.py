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
print("========== OPEN DATASETS ============")
train = DataSet('../data/SIMULATOR/simulator.csv', format='CSV')
train.clear_cache()
test = DataSet('../data/SIMULATOR/simulator.csv', format='CSV')
test.clear_cache()

mainslist = []
meterlist = []
buildings = 4
sample_period = 1
meter_key = 'dish washer'
for i in range(1,buildings):
    train_elec = train.buildings[i].elec
    train_meter = train_elec.submeters()[meter_key]
    train_mains_instance = train_meter.upstream_meter().instance()
    train_mains = train_elec.mains().all_meters()[train_mains_instance-1]

    mainslist += [train_mains]
    meterlist += [train_meter]

test_elec = test.buildings[buildings].elec
test_meter = test_elec.submeters()[meter_key]
test_mains_instance = test_meter.upstream_meter().instance()
test_mains = test_elec.mains().all_meters()[test_mains_instance-1]

results_dir = '../results/SIMULATOR-ACROSS-BUILDINGS-RNN-{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(results_dir)
train_logfile = os.path.join(results_dir, 'training.log')
val_logfile = os.path.join(results_dir, 'validation.log')

rnn = RNNDisaggregator(train_logfile, val_logfile)

start = time.time()
print("========== TRAIN ============")
epochs = 0
for i in range(1):
    rnn.train_across_buildings(mainslist, meterlist, epochs=1, sample_period=sample_period)
    epochs += 1
    rnn.export_model(os.path.join(results_dir, "SIMULATOR-RNN-h{}-{}-{}-{}epochs.h5".format(1, buildings, meter_key, epochs)))
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
rnn.disaggregate(test_mains, output, results_file, meterlist[0], sample_period=sample_period)
output.close()


headline = "========== RESULTS ============"
with open(results_file, "a") as text_file:
    text_file.write(headline + '\n')
print(headline)
result = DataSet(os.path.join(results_dir, disag_filename))
res_elec = result.buildings[buildings].elec
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
