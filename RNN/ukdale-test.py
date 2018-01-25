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
train = DataSet('../data/ukdale.h5')
train.clear_cache()
train.set_window(start="13-4-2013", end="31-7-2013")
validation = DataSet('../data/ukdale.h5')
validation.clear_cache()
validation.set_window(start="31-7-2013", end="31-8-2013")
test = DataSet('../data/ukdale.h5')
test.clear_cache()
test.set_window(start="13-9-2013", end="30-9-2013")

train_building = 1
validation_building = 1
test_building = 1
sample_period = 6
meter_key = 'kettle'

train_elec = train.buildings[train_building].elec
validation_elec = validation.buildings[validation_building].elec
test_elec = test.buildings[test_building].elec

train_meter = train_elec.submeters()[meter_key]
validation_meter = validation_elec.submeters()[meter_key]
test_meter = test_elec.submeters()[meter_key]

train_mains = train_elec.mains()
validation_mains = validation_elec.mains()
test_mains = test_elec.mains()

results_dir = '../results/UKDALE-RNN-{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(results_dir)
train_logfile = os.path.join(results_dir, 'training.log')
val_logfile = os.path.join(results_dir, 'validation.log')

rnn = RNNDisaggregator(train_logfile, val_logfile)

print("========== TRAIN ============")
epochs = 0
start = time.time()
for i in range(30):
    rnn.train(train_mains, train_meter, validation_mains, validation_meter, epochs=10, sample_period=sample_period)
    epochs += 10
    rnn.export_model(os.path.join(results_dir, "UKDALE-RNN-h{}-{}-{}epochs.h5".format(train_building, meter_key, epochs)))
    print("CHECKPOINT {}".format(epochs))
end = time.time()
print("Train =", end-start, "seconds.")

results_file = os.path.join(results_dir, 'results.txt')
headline = "========== DISAGGREGATE ============"
with open(results_file, "w") as text_file:
    text_file.write(headline + '\n')
print(headline)

# find best model (min validation loss)
validation = pd.read_csv(val_logfile)
epochs = np.array(validation.as_matrix()[:,0], dtype='int')
loss = np.array(validation.as_matrix()[:,1], dtype='float32')
argmin = np.argmin(loss)
best_epoch = epochs[argmin] + 1
rnn.import_model(os.path.join(results_dir, "UKDALE-RNN-h{}-{}-{}epochs.h5".format(train_building, meter_key, best_epoch)))
test_loss = rnn.evaluate(test_mains, test_meter, sample_period=sample_period)
line = 'Test loss: {}'.format(test_loss)
with open(results_file, "a") as text_file:
    text_file.write(line + '\n')
print(line)

disag_filename = 'disag-out.h5'
output = HDFDataStore(os.path.join(results_dir, disag_filename), 'w')
rnn.disaggregate(test_mains, output, results_file, train_meter, sample_period=sample_period)
output.close()

print("========== PLOTS ============")
# plot train and validation loss
plt.plot(epochs, loss, label='validation')
training = pd.read_csv(train_logfile)
epochs = np.array(training.as_matrix()[:,0], dtype='int')
loss = np.array(training.as_matrix()[:,1], dtype='float32')
plt.plot(epochs, loss, label='train')
plt.title('Test loss: {}'.format(test_loss))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(os.path.join(results_dir, 'loss.png'))
plt.close()

# plot predicted energy consumption
result = DataSet(os.path.join(results_dir, disag_filename))
res_elec = result.buildings[test_building].elec
predicted = res_elec[meter_key]
ground_truth = test_elec[meter_key]
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
predicted.plot(ax=ax1, plot_kwargs={'color': 'r', 'label': 'predicted'})
ground_truth.plot(ax=ax1, plot_kwargs={'color': 'b', 'label': 'ground truth'})
predicted.plot(ax=ax2, plot_kwargs={'color': 'r', 'label': 'predicted'}, plot_legend=False)
ground_truth.plot(ax=ax3, plot_kwargs={'color': 'b', 'label': 'ground truth'}, plot_legend=False)
ax1.set_title('Appliance: {}'.format(meter_key))
fig.legend()
fig.savefig(os.path.join(results_dir, 'predicted_vs_ground_truth.png'))

