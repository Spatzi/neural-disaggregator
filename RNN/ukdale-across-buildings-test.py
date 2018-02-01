from __future__ import print_function, division
import time
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from nilmtk import DataSet, HDFDataStore
from rnndisaggregator import RNNDisaggregator


IMPORT = True  # TODO: True if continue training

windows = {
    'train': ['13-4-2013', '31-7-2013'],
    'validation': ['31-7-2013', '31-8-2013'],
    'test': ['13-4-2013', '13-5-2013']
}

print("========== OPEN DATASETS ============")
train = DataSet('../data/ukdale.h5')
train.clear_cache()
train.set_window(start=windows['train'][0], end=windows['train'][1])
validation = DataSet('../data/ukdale.h5')
validation.clear_cache()
validation.set_window(start=windows['validation'][0], end=windows['validation'][1])
test = DataSet('../data/ukdale.h5')
test.clear_cache()
test.set_window(start=windows['test'][0], end=windows['test'][1])

train_mainslist = []
train_meterlist = []
val_mainslist = []
val_meterlist = []
train_buildings = [1,2]
val_buildings = [1,2]
test_building = 4
sample_period = 6
meter_key = 'kettle'
learning_rate = 1e-5

if IMPORT:
    results_dir = '../results/UKDALE-ACROSS-BUILDINGS-RNN-lr=1e-5-2018-01-27 18:32:42'  # TODO: insert directory name
else:
    results_dir = '../results/UKDALE-ACROSS-BUILDINGS-RNN-lr={}-{}'.format(learning_rate,
                                                                           datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(results_dir)
results_file = os.path.join(results_dir, 'results.txt')
with open(results_file, "w") as text_file:
    text_file.write('========== PARAMETERS ============' + '\n')
    text_file.write('train window: ({}, {})\n'.format(windows['train'][0], windows['train'][1]))
    text_file.write('validation window: ({}, {})\n'.format(windows['validation'][0], windows['validation'][1]))
    text_file.write('test window: ({}, {})\n'.format(windows['test'][0], windows['test'][1]))
    text_file.write('train buildings: {}\n'.format(train_buildings))
    text_file.write('validation buildings: {}\n'.format(val_buildings))
    text_file.write('test building: {}\n'.format(test_building))
    text_file.write('appliance: {}\n'.format(meter_key))
    text_file.write('sample period: {}\n'.format(sample_period))
    text_file.write('learning rate: {}\n'.format(learning_rate))

for i in train_buildings:
    train_elec = train.buildings[i].elec
    train_meter = train_elec.submeters()[meter_key]
    train_mains = train_elec.mains()

    train_mainslist += [train_mains]
    train_meterlist += [train_meter]

for i in val_buildings:
    val_elec = validation.buildings[i].elec
    val_meter = val_elec.submeters()[meter_key]
    val_mains = val_elec.mains()

    val_mainslist += [val_mains]
    val_meterlist += [val_meter]

test_elec = test.buildings[test_building].elec
test_meter = test_elec.submeters()[meter_key]
test_mains = test_elec.mains()

train_logfile = os.path.join(results_dir, 'training.log')
val_logfile = os.path.join(results_dir, 'validation.log')

if IMPORT:
    rnn = RNNDisaggregator(train_logfile, val_logfile, learning_rate, init=False)
    rnn.import_model(os.path.join(results_dir, "UKDALE-RNN-kettle-380epochs.h5"))  # TODO: insert last model name
else:
    rnn = RNNDisaggregator(train_logfile, val_logfile, learning_rate)

start = time.time()
print("========== TRAIN ============")
epochs = 380  # TODO: update according to the last model if IMPORT = True
for i in range(22):
    rnn.train_across_buildings(train_mainslist, train_meterlist, val_mainslist, val_meterlist, epochs=10,
                               sample_period=sample_period)
    epochs += 10
    rnn.export_model(os.path.join(results_dir, "UKDALE-RNN-{}-{}epochs.h5".format(meter_key, epochs)))
    print("CHECKPOINT {}".format(epochs))
end = time.time()
print("Train =", end-start, "seconds.")

results_file = os.path.join(results_dir, 'results.txt')
headline = "========== DISAGGREGATE ============"
with open(results_file, "a") as text_file:
    text_file.write(headline + '\n')
print(headline)

# find best model (min validation loss)
validation = pd.read_csv(val_logfile)
epochs = np.array(validation.as_matrix()[:,0], dtype='int')
loss = np.array(validation.as_matrix()[:,1], dtype='float32')
argmin = np.argmin(loss)
best_epoch = epochs[argmin] + 1
rnn.import_model(os.path.join(results_dir, "UKDALE-RNN-{}-{}epochs.h5".format(meter_key, best_epoch)))
test_loss = rnn.evaluate(test_mains, test_meter, sample_period=sample_period)
line = 'Test loss: {}'.format(test_loss)
with open(results_file, "a") as text_file:
    text_file.write(line + '\n')
print(line)

disag_filename = 'disag-out.h5'
output = HDFDataStore(os.path.join(results_dir, disag_filename), 'w')
rnn.disaggregate(test_mains, output, results_file, train_meterlist[0], sample_period=sample_period)
output.close()

print("========== PLOTS ============")
# plot train and validation loss
plt.plot(epochs, loss, label='validation')
training = pd.read_csv(train_logfile)
epochs = np.array(training.as_matrix()[1:,0], dtype='int')
loss = np.array(training.as_matrix()[1:,1], dtype='float32')
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
