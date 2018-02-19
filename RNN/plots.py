import os
import plotly
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from nilmtk import DataSet, HDFDataStore
from rnndisaggregator import RNNDisaggregator


def generate_vertices():
    train = DataSet('../data/ukdale.h5')
    train.clear_cache()
    train.set_window(start="13-4-2013", end="31-7-2013")
    test = DataSet('../data/ukdale.h5')
    test.clear_cache()
    test.set_window(start='7-2-2014 08:00:00', end='7-3-2014')

    train_building = 1
    test_building = 5
    sample_period = 6
    meter_key = 'kettle'
    learning_rate = 1e-5

    train_elec = train.buildings[train_building].elec
    test_elec = test.buildings[test_building].elec

    train_meter = train_elec.submeters()[meter_key]
    test_mains = test_elec.mains()

    results_dir = '../results/UKDALE-ACROSS-BUILDINGS-RNN-lr=1e-05-2018-02-03-11-48-12'
    train_logfile = os.path.join(results_dir, 'training.log')
    val_logfile = os.path.join(results_dir, 'validation.log')
    rnn = RNNDisaggregator(train_logfile, val_logfile, learning_rate, init=False)

    verts = []
    zs = []  # epochs
    for z in np.arange(10, 341, 10):

        # disaggregate model
        model = 'UKDALE-RNN-kettle-{}epochs.h5'.format(z)
        rnn.import_model(os.path.join(results_dir, model))
        disag_filename = 'disag-out-{}epochs.h5'.format(z)
        output = HDFDataStore(os.path.join(results_dir, disag_filename), 'w')
        results_file = os.path.join(results_dir, 'results-{}epochs.txt'.format(z))
        rnn.disaggregate(test_mains, output, results_file, train_meter, sample_period=sample_period)
        os.remove(results_file)
        output.close()

        # get predicted curve for epoch=z
        result = DataSet(os.path.join(results_dir, disag_filename))
        res_elec = result.buildings[test_building].elec
        os.remove(os.path.join(results_dir, disag_filename))
        predicted = res_elec[meter_key]
        predicted = predicted.power_series(sample_period=sample_period)
        predicted = next(predicted)
        predicted.fillna(0, inplace=True)
        ys = np.array(predicted)  # power
        xs = np.arange(ys.shape[0])  # timestamps

        verts.append(list(zip(xs, ys)))  # add list of x-y-coordinates
        zs.append(z)

    ground_truth = test_elec[meter_key]
    ground_truth = ground_truth.power_series(sample_period=sample_period)
    ground_truth = next(ground_truth)
    ground_truth.fillna(0, inplace=True)
    ys = np.array(ground_truth)  # power
    xs = np.arange(ys.shape[0])  # timestamps

    verts.append(list(zip(xs, ys)))  # add list of x-y-coordinates
    zs.append(350)

    zs = np.asarray(zs)

    for i in range(len(verts)):
        verts[i].insert(0, [0, np.array([0])])
        verts[i].append([len(verts[i]), np.array([0])])

    pickle.dump(verts, open(os.path.join(results_dir, 'vertices.pkl'), 'wb'))
    pickle.dump(zs, open(os.path.join(results_dir, 'zs.pkl'), 'wb'))
    pickle.dump(ys, open(os.path.join(results_dir, 'ys.pkl'), 'wb'))


def plot_prediction_over_epochs_plt():
    generate_vertices()
    results_dir = '../results/UKDALE-RNN-lr=1e-5-2018-01-26 14:33:59'
    verts = pickle.load(open(os.path.join(results_dir, 'vertices.pkl'), 'rb'))
    zs = pickle.load(open(os.path.join(results_dir, 'zs.pkl'), 'rb'))
    ys = pickle.load(open(os.path.join(results_dir, 'ys.pkl'), 'rb'))

    fig = plt.figure(figsize=(18, 8))
    ax = fig.gca(projection='3d')

    poly = PolyCollection(verts[::], facecolors='w')
    poly.set_edgecolor((0, 0, 0, .5))
    poly.set_facecolor((.9, .9, 1, 0.3))
    ax.add_collection3d(poly, zs=zs[::], zdir='y')

    ax.set_xlabel('timestamps')
    ax.set_xlim3d(0, ys.shape[0])
    ax.set_ylabel('epochs')
    ax.set_ylim3d(0, 320)
    ax.set_zlabel('power')
    ax.set_zlim3d(0, 2000)
    ax.view_init(-40,-94)

    # plt.savefig(os.path.join(results_dir, 'prediction_over_epochs.png'))
    plt.show()
    print('ok')


def plot_prediction_over_epochs_ploty():
    train = DataSet('../data/ukdale.h5')
    train.clear_cache()
    train.set_window(start="13-4-2013", end="31-7-2013")
    test = DataSet('../data/ukdale.h5')
    test.clear_cache()
    test.set_window(start="23-7-2014 10:00:00", end="23-7-2014 11:00:00")

    train_building = 1
    test_building = 5
    sample_period = 6
    meter_key = 'kettle'
    learning_rate = 1e-5

    train_elec = train.buildings[train_building].elec
    test_elec = test.buildings[test_building].elec

    train_meter = train_elec.submeters()[meter_key]
    test_mains = test_elec.mains()

    results_dir = '../results/UKDALE-ACROSS-BUILDINGS-RNN-lr=1e-05-2018-02-03-11-48-12'
    train_logfile = os.path.join(results_dir, 'training.log')
    val_logfile = os.path.join(results_dir, 'validation.log')
    rnn = RNNDisaggregator(train_logfile, val_logfile, learning_rate, init=False)

    data = []

    for i in range(10, 401, 10):
        # disaggregate model
        model = 'UKDALE-RNN-kettle-{}epochs.h5'.format(i)
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
        power = predicted.tolist()
        length = len(power)
        timestamps = list(range(length))

        x = []
        y = []
        z = []
        ci = int(255 / 420 * i)  # ci = "color index"
        for j in range(length):
            x.append([timestamps[j], timestamps[j]])  # timestamps
            y.append([i, i + 5])  # epochs
            z.append([power[j], power[j]])  # power
        data.append(dict(
            z=z,
            x=x,
            y=y,
            colorscale=[[i, 'rgb(%d,%d,255)' % (ci, ci)] for i in np.arange(0, 1.1, 0.1)],
            showscale=False,
            type='surface',
        ))

    # plot ground truth curve as the last curve
    ground_truth = test_elec[meter_key]
    ground_truth = ground_truth.power_series(sample_period=sample_period)
    ground_truth = next(ground_truth)
    ground_truth.fillna(0, inplace=True)
    power = ground_truth.tolist()
    length = len(power)
    timestamps = list(range(length))

    i = 410
    x = []
    y = []
    z = []
    ci = int(255 / 410 * i)  # ci = "color index"
    for j in range(length):
        x.append([timestamps[j], timestamps[j]])  # timestamps
        y.append([i, i + 5])  # epochs
        z.append([power[j], power[j]])  # power
    data.append(dict(
        z=z,
        x=x,
        y=y,
        colorscale=[[i, 'rgb(%d,%d,255)' % (ci, ci)] for i in np.arange(0, 1.1, 0.1)],
        showscale=False,
        type='surface',
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

    # windows = {
    #     'train': [["21-5-2013", "21-8-2013"]],
    #     'validation': ["9-10-2013", "10-10-2013"],
    #     'test': ["11-1-2013", "15-11-2013"]
    # }
    windows = {
        'train': [["21-5-2013", "9-1-2013"]],
        'validation': ["9-1-2013", "29-9-2013"],
        'test': ["15-7-2014", "15-8-2014"] #5
    }

    train = []
    for window in windows['train']:
        t = DataSet('../data/ukdale.h5')
        t.clear_cache()
        t.set_window(start=window[0], end=window[1])
        train += [t]

    validation = DataSet('../data/ukdale.h5')
    validation.clear_cache()
    validation.set_window(start=windows['validation'][0], end=windows['validation'][1])
    test = DataSet('../data/ukdale.h5')
    test.clear_cache()
    test.set_window(start=windows['test'][0], end=windows['test'][1])

    train_buildings = [2]
    val_buildings = [2]
    test_building = 5
    sample_period = 6
    meter_key = 'dish washer'

    train_meterlist = []
    val_meterlist = []

    for i, b in enumerate(train_buildings):
        train_elec = train[i].buildings[b].elec
        train_meterlist += [train_elec.submeters()[meter_key]]

    for i in val_buildings:
        val_elec = validation.buildings[i].elec
        val_meterlist += [val_elec.submeters()[meter_key]]

    test_elec = test.buildings[test_building].elec
    test_meter = test_elec.submeters()[meter_key]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=False, sharey=True)
    fig.set_size_inches(15.5, 10.5)
    train_meterlist[0].plot(ax=ax1, plot_kwargs={'color': 'g', 'label': 'Train set - building 1'}, plot_legend=False)
    # train_meterlist[1].plot(ax=ax2, plot_kwargs={'color': 'b', 'label': 'Train set - building 2'}, plot_legend=False)
    val_meterlist[0].plot(ax=ax3, plot_kwargs={'color': 'y', 'label': 'Validation set - building 1'}, plot_legend=False)
    test_meter.plot(ax=ax4, plot_kwargs={'color': 'r', 'label': 'Test set - building 1'}, plot_legend=False)
    ax1.set_title('Appliance: {}'.format(meter_key))
    fig.legend()
    plt.savefig('datasets.png')


def plot_zoomed_new_predicted_energy_consumption():
    """
    New prediction.
    """
    train = DataSet('../data/ukdale.h5')
    train.clear_cache()
    train.set_window(start="2-1-2014", end="15-5-2014")
    test = DataSet('../data/ukdale.h5')
    test.clear_cache()
    test.set_window(start='25-6-2014 20:00:00', end='26-6-2014 04:00:00')

    train_building = 1
    test_building = 1
    sample_period = 6
    meter_key = 'fridge'
    learning_rate = 1e-5
    best_epoch = 560

    train_elec = train.buildings[train_building].elec
    test_elec = test.buildings[test_building].elec

    train_meter = train_elec.submeters()[meter_key]
    test_mains = test_elec.mains()

    results_dir = '../results/UKDALE-RNN-lr=1e-05-2018-01-28-12-01-34'
    train_logfile = os.path.join(results_dir, 'training.log')
    val_logfile = os.path.join(results_dir, 'validation.log')
    rnn = RNNDisaggregator(train_logfile, val_logfile, learning_rate, init=False)

    model = 'UKDALE-RNN-fridge-{}epochs.h5'.format(best_epoch)
    rnn.import_model(os.path.join(results_dir, model))
    disag_filename = 'disag-out-{}epochs.h5'.format(best_epoch)
    output = HDFDataStore(os.path.join(results_dir, disag_filename), 'w')
    results_file = os.path.join(results_dir, 'results-{}epochs.txt'.format(best_epoch))
    rnn.disaggregate(test_mains, output, results_file, train_meter, sample_period=sample_period)
    os.remove(results_file)
    output.close()

    # get predicted curve for the best epoch
    result = DataSet(os.path.join(results_dir, disag_filename))
    res_elec = result.buildings[test_building].elec
    os.remove(os.path.join(results_dir, disag_filename))
    predicted = res_elec[meter_key]
    predicted = predicted.power_series(sample_period=sample_period)
    predicted = next(predicted)
    predicted.fillna(0, inplace=True)
    y1 = np.array(predicted)  # power
    x1 = np.arange(y1.shape[0])  # timestamps

    ground_truth = test_elec[meter_key]
    ground_truth = ground_truth.power_series(sample_period=sample_period)
    ground_truth = next(ground_truth)
    ground_truth.fillna(0, inplace=True)
    y2 = np.array(ground_truth)  # power
    x2 = np.arange(y2.shape[0])  # timestamps

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.plot(x1, y1, color='r', label='predicted')
    ax1.plot(x2, y2, color='b', label='ground truth')
    ax2.plot(x1, y1, color='r')
    ax3.plot(x2, y2, color='b')
    ax1.set_title('Appliance: {}'.format(meter_key))
    # plt.xticks(np.arange(0,x2.shape[0]+1,90), ('15-9-2013 15:30', '15:40', '15:50', '16:00', '16:10', '16:20'))
    fig.legend()
    fig.savefig(os.path.join(results_dir, 'zoomed_new_predicted_vs_ground_truth.png'))


def plot_zoomed_original_predicted_energy_consumption():
    """
    Original prediction.
    """
    test = DataSet('../data/ukdale.h5')
    test.clear_cache()
    test.set_window(start='29-9-2013', end='10-10-2013')

    test_building = 2
    sample_period = 6
    meter_key = 'dish washer'

    test_elec = test.buildings[test_building].elec

    results_dir = '../results/UKDALE-RNN-lr=1e-05-2018-02-16-12-29-50'
    disag_filename = 'disag-out.h5'

    # get predicted curve for the best epoch
    result = DataSet(os.path.join(results_dir, disag_filename))
    res_elec = result.buildings[test_building].elec
    predicted = res_elec[meter_key]
    predicted = predicted.power_series(sample_period=sample_period)
    predicted = next(predicted)
    predicted.fillna(0, inplace=True)
    y1 = np.array(predicted)  # power
    x1 = np.arange(y1.shape[0])  # timestamps
    x1 = x1[90000:11000]
    y1 = y1[90000:11000]

    ground_truth = test_elec[meter_key]
    ground_truth = ground_truth.power_series(sample_period=sample_period)
    ground_truth = next(ground_truth)
    ground_truth.fillna(0, inplace=True)
    y2 = np.array(ground_truth)  # power
    x2 = np.arange(y2.shape[0])  # timestamps
    x2 = x2[90000:11000]
    y2 = y2[90000:11000]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.plot(x1, y1, color='r', label='predicted')
    ax1.plot(x2, y2, color='b', label='ground truth')
    ax2.plot(x1, y1, color='r')
    ax3.plot(x2, y2, color='b')
    ax1.set_title('Appliance: {}'.format(meter_key))
    # plt.xticks(np.arange(0,x2.shape[0]+1,90), ('15-9-2013 15:30', '15:40', '15:50', '16:00', '16:10', '16:20'))
    fig.legend()
    fig.savefig(os.path.join(results_dir, 'zoomed_original_predicted_vs_ground_truth.png'))


if __name__ == "__main__":
    plot_zoomed_original_predicted_energy_consumption()