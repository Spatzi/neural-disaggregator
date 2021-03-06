from __future__ import print_function, division
import time
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from nilmtk import DataSet, HDFDataStore
from rnndisaggregator import RNNDisaggregator
from plots import plot_loss


IMPORT = False  # TODO: True if continue training

windows = {
    'train': [['13-4-2013', '13-5-2013'], ['13-6-2013', '13-7-2013'], ['13-5-2013', '13-6-2013'],
              ['13-7-2014', '31-7-2014']],
    'validation': [['13-5-2013', '20-5-2013'], ['13-7-2013', '20-7-2013'], ['13-4-2013', '20-4-2013'],
                   ['7-6-2014', '13-7-2014']],
    'test': ['30-6-2013', '15-7-2013']
}

train = []
print("========== OPEN DATASETS ============")
for window in windows['train']:
    t = DataSet('../data/ukdale.h5')
    t.clear_cache()
    t.set_window(start=window[0], end=window[1])
    train += [t]

validation = []
for window in windows['validation']:
    v = DataSet('../data/ukdale.h5')
    v.clear_cache()
    v.set_window(start=window[0], end=window[1])
    validation += [v]

test = DataSet('../data/ukdale.h5')
test.clear_cache()
test.set_window(start=windows['test'][0], end=windows['test'][1])

train_mainslist = []
train_meterlist = []
val_mainslist = []
val_meterlist = []
train_buildings = [1,2,4,5]
val_buildings = [1,2,4,5]
test_building = 1
sample_period = 6
meter_key = 'kettle'
learning_rate = 1e-5

if IMPORT:
    results_dir = '../results/UKDALE-ACROSS-BUILDINGS-RNN-lr=1e-05-2018-02-20-14-24-46'  # TODO: insert directory name
else:
    results_dir = '../results/UKDALE-ACROSS-BUILDINGS-RNN-lr={}-{}'.format(learning_rate,
                                                                           datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(results_dir)
results_file = os.path.join(results_dir, 'results.txt')
with open(results_file, "w") as text_file:
    text_file.write('========== PARAMETERS ============' + '\n')
    for window in windows['train']:
        text_file.write('train window: ({}, {})\n'.format(window[0], window[1]))
    for window in windows['validation']:
        text_file.write('validation window: ({}, {})\n'.format(window[0], window[1]))
    text_file.write('test window: ({}, {})\n'.format(windows['test'][0], windows['test'][1]))
    text_file.write('train buildings: {}\n'.format(train_buildings))
    text_file.write('validation buildings: {}\n'.format(val_buildings))
    text_file.write('test building: {}\n'.format(test_building))
    text_file.write('appliance: {}\n'.format(meter_key))
    text_file.write('sample period: {}\n'.format(sample_period))
    text_file.write('learning rate: {}\n'.format(learning_rate))

for i, b in enumerate(train_buildings):
    train_elec = train[i].buildings[b].elec
    train_meter = train_elec.submeters()[meter_key]
    train_mains = train_elec.mains()

    train_mainslist += [train_mains]
    train_meterlist += [train_meter]

for i, b in enumerate(val_buildings):
    val_elec = validation[i].buildings[b].elec
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
    rnn.import_model(os.path.join(results_dir, "UKDALE-RNN-kettle-200epochs.h5"))  # TODO: insert last model name
else:
    rnn = RNNDisaggregator(train_logfile, val_logfile, learning_rate)

start = time.time()
print("========== TRAIN ============")
epochs = 0  # TODO: update according to the last model if IMPORT = True
for i in range(30):
    rnn.train_across_buildings(train_mainslist, train_meterlist, val_mainslist, val_meterlist, epochs=10,
                               sample_period=sample_period)
    epochs += 10
    rnn.export_model(os.path.join(results_dir, "UKDALE-RNN-{}-{}epochs.h5".format(meter_key, epochs)))
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
# plot train, validation and test loss
plot_loss(train_logfile, val_logfile, results_dir, best_epoch, test_loss)

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
