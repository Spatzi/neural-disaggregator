from __future__ import print_function, division

import time
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from nilmtk import DataSet, HDFDataStore
from rnndisaggregator import RNNDisaggregator
from plots import plot_loss


IMPORT = True

windows = {
        'train': ["13-4-2013", "20-4-2013"],
        'validation': ["20-4-2013", "22-4-2013"],
        'test': ["22-4-2013", "23-4-2013"]
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

train_building = 1
validation_building = 1
test_building = 1
sample_period = 6
meter_keys = ['kettle', 'microwave']
learning_rate = 1e-5

if IMPORT:
    results_dir = '../results/UKDALE-RNN-lr=1e-05-2018-02-19-19-23-32'  # TODO: insert directory name
else:
    results_dir = '../results/UKDALE-RNN-lr={}-{}'.format(learning_rate, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(results_dir)
results_file = os.path.join(results_dir, 'results.txt')
with open(results_file, "w") as text_file:
    text_file.write('========== PARAMETERS ============' + '\n')
    text_file.write('USE SLIDING WINDOW' + '\n')
    text_file.write('train window: ({}, {})\n'.format(windows['train'][0], windows['train'][1]))
    text_file.write('validation window: ({}, {})\n'.format(windows['validation'][0], windows['validation'][1]))
    text_file.write('test window: ({}, {})\n'.format(windows['test'][0], windows['test'][1]))
    text_file.write('train building: {}\n'.format(train_building))
    text_file.write('validation building: {}\n'.format(validation_building))
    text_file.write('test building: {}\n'.format(test_building))
    text_file.write('appliances: {}\n'.format(meter_keys))
    text_file.write('sample period: {}\n'.format(sample_period))
    text_file.write('learning rate: {}\n'.format(learning_rate))

train_elec = train.buildings[train_building].elec
validation_elec = validation.buildings[validation_building].elec
test_elec = test.buildings[test_building].elec

train_meters = []
validation_meters = []
test_meters = []
for key in meter_keys:
    train_meters += [train_elec.submeters()[key]]
    validation_meters += [validation_elec.submeters()[key]]
    test_meters += [test_elec.submeters()[key]]

train_mains = train_elec.mains()
validation_mains = validation_elec.mains()
test_mains = test_elec.mains()

train_logfile = os.path.join(results_dir, 'training.log')
val_logfile = os.path.join(results_dir, 'validation.log')

if IMPORT:
    rnn = RNNDisaggregator(train_logfile, val_logfile, learning_rate, init=False)
    rnn.import_model(os.path.join(results_dir, "UKDALE-RNN-['kettle', 'microwave']-1epochs.h5"))  # TODO: insert last model name
else:
    rnn = RNNDisaggregator(train_logfile, val_logfile, learning_rate)

print("========== TRAIN ============")
epochs = 0  # TODO: update according to the last model if IMPORT = True
start = time.time()
for i in range(0):
    rnn.train(train_mains, train_meters, validation_mains, validation_meters, epochs=1, sample_period=sample_period)
    epochs += 1
    rnn.export_model(os.path.join(results_dir, "UKDALE-RNN-{}-{}epochs.h5".format(meter_keys, epochs)))
    plot_loss(train_logfile, val_logfile, results_dir)
    print("CHECKPOINT {}".format(epochs))
end = time.time()
print("Train =", end-start, "seconds.")

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
rnn.import_model(os.path.join(results_dir, "UKDALE-RNN-{}-{}epochs.h5".format(meter_keys, best_epoch)))
test_loss = rnn.evaluate(test_mains, test_meters, sample_period=sample_period)
line = 'Test loss: {}'.format(test_loss)
with open(results_file, "a") as text_file:
    text_file.write(line + '\n')
print(line)

disag_filename = 'disag-out.h5'
output = HDFDataStore(os.path.join(results_dir, disag_filename), 'w')
rnn.disaggregate(test_mains, output, results_file, train_meters, sample_period=sample_period)
output.close()

print("========== PLOTS ============")
# plot train, validation and test loss
plot_loss(train_logfile, val_logfile, results_dir, best_epoch, test_loss)

# plot predicted energy consumption
result = DataSet(os.path.join(results_dir, disag_filename))
res_elec = result.buildings[test_building].elec
for key in meter_keys:
    predicted = res_elec[key]
    ground_truth = test_elec[key]
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    predicted.plot(ax=ax1, plot_kwargs={'color': 'r', 'label': 'predicted'})
    ground_truth.plot(ax=ax1, plot_kwargs={'color': 'b', 'label': 'ground truth'})
    predicted.plot(ax=ax2, plot_kwargs={'color': 'r', 'label': 'predicted'}, plot_legend=False)
    ground_truth.plot(ax=ax3, plot_kwargs={'color': 'b', 'label': 'ground truth'}, plot_legend=False)
    ax1.set_title('Appliance: {}'.format(key))
    fig.legend()
    fig.savefig(os.path.join(results_dir, 'predicted_vs_ground_truth_{}.png'.format(key)))
