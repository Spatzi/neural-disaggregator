import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from nilmtk import DataSet, HDFDataStore
from rnndisaggregator import RNNDisaggregator

import plotly.plotly as py
import plotly


def plot_prediction_over_epochs():
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

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # for i in range(10, 21, 10):
    #     # disaggregate model
    #     model = 'UKDALE-RNN-h1-kettle-{}epochs.h5'.format(i)
    #     rnn.import_model(os.path.join(results_dir, model))
    #     disag_filename = 'disag-out-{}epochs.h5'.format(i)
    #     output = HDFDataStore(os.path.join(results_dir, disag_filename), 'w')
    #     results_file = os.path.join(results_dir, 'results-{}epochs.txt'.format(i))
    #     rnn.disaggregate(test_mains, output, results_file, train_meter, sample_period=sample_period)
    #     os.remove(results_file)
    #     output.close()
    #
    #     # plot predicted curve for epoch=i
    #     result = DataSet(os.path.join(results_dir, disag_filename))
    #     res_elec = result.buildings[test_building].elec
    #     os.remove(os.path.join(results_dir, disag_filename))
    #     predicted = res_elec[meter_key]
    #     predicted = predicted.power_series(sample_period=sample_period)
    #     predicted = next(predicted)
    #     predicted.fillna(0, inplace=True)
    #     # timestamps = np.array(predicted.keys())
    #     power = np.array(predicted)
    #     timestamps = np.array(range(power.shape[0]))
    #     ax.plot(timestamps, power, zs=i, zdir='z', color='b', alpha=0.3)
    #
    # # plot ground truth curve as the last curve
    # ground_truth = test_elec[meter_key]
    # ground_truth = ground_truth.power_series(sample_period=sample_period)
    # ground_truth = next(ground_truth)
    # ground_truth.fillna(0, inplace=True)
    # # timestamps = np.array(ground_truth.keys())
    # power = np.array(ground_truth)
    # timestamps = np.array(range(power.shape[0]))
    # ax.plot(timestamps, power, zs=110, zdir='z', color='r')
    # ax.fill_between(timestamps, 0, power)
    # ax.set_xlabel('timestamps')
    # ax.set_ylabel('power')
    # ax.set_zlabel('epochs')
    # plt.show()


    fill_colors = ['rgba(0,255,0,0.1)', 'rgba(255,0,0,1)']
    data = []

    for i in range(10, 21, 10):
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
        # power = np.array(predicted)
        # timestamps = np.array(range(power.shape[0]))
        power = predicted.tolist()
        length = len(power)
        timestamps = list(range(length))
        epochs = [i] * length

        data.append(dict(
            type='scatter3d',
            mode='lines',
            x=timestamps,
            y=epochs,
            z=power,
            name='',
            surfaceaxis=1,  # add a surface axis ('1' refers to axes[1] i.e. the y-axis)
            surfacecolor=fill_colors[0],
            line=dict(
                color='black',
                width=4
            ),
        ))

    # plot ground truth curve as the last curve
    ground_truth = test_elec[meter_key]
    ground_truth = ground_truth.power_series(sample_period=sample_period)
    ground_truth = next(ground_truth)
    ground_truth.fillna(0, inplace=True)
    # timestamps = np.array(ground_truth.keys())
    power = ground_truth.tolist()
    length = len(power)
    timestamps = list(range(length))
    epochs = [30] * length

    data.append(dict(
        type='scatter3d',
        mode='lines',
        x=timestamps,
        y=epochs,
        z=power,
        name='',
        surfaceaxis=1,  # add a surface axis ('1' refers to axes[1] i.e. the y-axis)
        surfacecolor=fill_colors[1],
        line=dict(
            color='black',
            width=4
        ),
    ))

    layout = dict(
        title='prediction over epochs',
        showlegend=False,
        scene=dict(
            xaxis=dict(title='timestamps'),
            yaxis=dict(title='epochs'),
            zaxis=dict(title='power'),
            camera=dict(
                eye=dict(x=-1.7, y=-1.7, z=0.5)
            )
        )
    )

    fig = dict(data=data, layout=layout)

    # IPython notebook
    # py.iplot(fig, filename='filled-3d-lines')

    plotly.offline.plot(fig, filename='filled-3d-lines')


def plot_loss(train_logfile, val_logfile, results_dir, best_epoch=None, test_loss=None):
    """
    Plot train and validation loss. In addition, plot test loss for the best epoch (if available).
    :param train_logfile: Training loss loggin.
    :param val_logfile: Validation loss loggin.
    :param results_dir: The directory to save the plot.
    :param best_epoch: The epoch with the minimum validation loss. None if not available.
    :param test_loss: The test loss in the best epoch. None if not available.
    """

    validation = pd.read_csv(val_logfile)
    epochs = np.array(validation.as_matrix()[:, 0], dtype='int')
    loss = np.array(validation.as_matrix()[:, 1], dtype='float32')
    plt.plot(epochs, loss, label='validation')
    training = pd.read_csv(train_logfile)
    epochs = np.array(training.as_matrix()[1:, 0], dtype='int')
    loss = np.array(training.as_matrix()[1:, 1], dtype='float32')
    plt.plot(epochs, loss, label='train')

    if best_epoch and test_loss:
        plt.plot([best_epoch - 1], [test_loss], 'ro', label='test')
        plt.title('Test loss: {}'.format(test_loss))

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss.png'))
    plt.close()


def plot_datasets_meter():

    windows = {
        'train': ['13-4-2013', '31-7-2013'],
        'validation': ['13-4-2013', '13-6-2013'],
        'test': ['30-6-2014', '31-7-2014']
    }

    train = DataSet('../data/ukdale.h5')
    train.clear_cache()
    train.set_window(start=windows['train'][0], end=windows['train'][1])
    validation = DataSet('../data/ukdale.h5')
    validation.clear_cache()
    validation.set_window(start=windows['validation'][0], end=windows['validation'][1])
    test = DataSet('../data/ukdale.h5')
    test.clear_cache()
    test.set_window(start=windows['test'][0], end=windows['test'][1])

    train_buildings = [1, 2]
    val_buildings = [4]
    test_building = 5
    sample_period = 6
    meter_key = 'kettle'

    train_meterlist = []
    val_meterlist = []

    for i in val_buildings:
        val_elec = validation.buildings[i].elec
        val_meterlist += [val_elec.submeters()[meter_key]]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    val_meterlist[0].plot(ax=ax2, plot_kwargs={'color': 'g'}, plot_legend=False)
    ax2.set_title('Validation set - building 4')
    fig.legend()
    fig.savefig('val.png')

    test_elec = test.buildings[test_building].elec
    test_meter = test_elec.submeters()[meter_key]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    test_meter.plot(ax=ax2, plot_kwargs={'color': 'g'}, plot_legend=False)
    ax2.set_title('Test set - building 5')
    fig.legend()
    fig.savefig('test.png')

    for i in train_buildings:
        train_elec = train.buildings[i].elec
        train_meterlist += [train_elec.submeters()[meter_key]]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    train_meterlist[0].plot(ax=ax1, plot_kwargs={'color': 'g'}, plot_legend=False)
    train_meterlist[1].plot(ax=ax2, plot_kwargs={'color': 'g'}, plot_legend=False)
    ax1.set_title('Train set - building 1')
    ax2.set_title('Train set - building 2')
    fig.legend()
    fig.savefig('train.png')


if __name__ == "__main__":
    plot_datasets_meter()