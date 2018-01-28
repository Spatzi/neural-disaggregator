import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from nilmtk import DataSet, HDFDataStore
from rnndisaggregator import RNNDisaggregator

train = DataSet('../data/ukdale.h5')
train.clear_cache()
train.set_window(start="18-4-2013", end="14-5-2013")
test = DataSet('../data/ukdale.h5')
test.clear_cache()
test.set_window(start="21-5-2013", end="24-5-2013")

train_building = 1
test_building = 1
sample_period = 6
meter_key = 'kettle'
learning_rate = 1e-3

train_elec = train.buildings[train_building].elec
test_elec = test.buildings[test_building].elec

train_meter = train_elec.submeters()[meter_key]
test_mains = test_elec.mains()

results_dir = '../results/UKDALE-RNN-2018-01-12 00:05:42'
train_logfile = os.path.join(results_dir, 'training.log')
val_logfile = os.path.join(results_dir, 'validation.log')
rnn = RNNDisaggregator(train_logfile, val_logfile, learning_rate, init=False)

fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(10, 101, 10):
    # disaggregate model
    model = 'UKDALE-RNN-h1-kettle-{}epochs.h5'.format(i)
    rnn.import_model(os.path.join(results_dir, model))
    disag_filename = 'disag-out-{}epochs.h5'.format(i)
    output = HDFDataStore(os.path.join(results_dir, disag_filename), 'w')
    results_file = os.path.join(results_dir, 'results-{}epochs.txt'.format(i))
    rnn.disaggregate(test_mains, output, results_file, train_meter, sample_period=sample_period)
    os.remove(results_file)
    output.close()

    # plot predicted curve for epoch=i
    result = DataSet(os.path.join(results_dir, disag_filename))
    res_elec = result.buildings[test_building].elec
    os.remove(os.path.join(results_dir, disag_filename))
    predicted = res_elec[meter_key]
    predicted = predicted.power_series(sample_period=sample_period)
    predicted = next(predicted)
    predicted.fillna(0, inplace=True)
    # timestamps = np.array(predicted.keys())
    power = np.array(predicted)
    timestamps = np.array(range(power.shape[0]))
    ax.plot(timestamps, power, zs=i, zdir='z', color='b')

# plot ground truth curve as the last curve
ground_truth = test_elec[meter_key]
ground_truth = ground_truth.power_series(sample_period=sample_period)
ground_truth = next(ground_truth)
ground_truth.fillna(0, inplace=True)
# timestamps = np.array(ground_truth.keys())
power = np.array(ground_truth)
timestamps = np.array(range(power.shape[0]))
ax.plot(timestamps, power, zs=110, zdir='z', color='r')
ax.set_xlabel('timestamps')
ax.set_ylabel('power')
ax.set_zlabel('epochs')
plt.show()

