from __future__ import print_function, division
from warnings import warn, filterwarnings

from matplotlib import rcParams
import matplotlib.pyplot as plt
import csv

import random
import sys
import pandas as pd
import numpy as np
import h5py

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Bidirectional, Dropout
from keras.utils import plot_model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class RNNDisaggregator(Disaggregator):
    '''Attempt to create a RNN Disaggregator

    Attributes
    ----------
    model : keras Sequential model
    mmax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self, train_logfile, val_logfile):
        '''Initialize disaggregator
        '''
        self.MODEL_NAME = "LSTM"
        self.mmax = None
        self.MIN_CHUNK_LENGTH = 100
        self.model = self._create_model()
        self.total_epochs = 0
        self.train_logfile = train_logfile
        self.val_logfile = val_logfile

        self.init_logfile(train_logfile)
        self.init_logfile(val_logfile)

    def train(self, train_mains, train_meter, validation_mains, validation_meter, epochs=1, batch_size=128, **load_kwargs):
        '''Train

        Parameters
        ----------
        train_mains : a nilmtk.ElecMeter object for the aggregate data
        train_meter : a nilmtk.ElecMeter object for the train_meter data
        epochs : number of epochs to train
        batch_size : size of batch used for training
        **load_kwargs : keyword arguments passed to `train_meter.power_series()`
        '''

        train_main_power_series = train_mains.power_series(**load_kwargs)
        train_meter_power_series = train_meter.power_series(**load_kwargs)
        val_main_power_series = validation_mains.power_series(**load_kwargs)
        val_meter_power_series = validation_meter.power_series(**load_kwargs)

        # Train chunks
        run = True
        train_mainchunk = next(train_main_power_series)
        train_meterchunk = next(train_meter_power_series)
        val_mainchunk = next(val_main_power_series)
        val_meterchunk = next(val_meter_power_series)
        if self.mmax is None:
            self.mmax = train_mainchunk.max()

        while run:
            train_mainchunk = self._normalize(train_mainchunk, self.mmax)
            train_meterchunk = self._normalize(train_meterchunk, self.mmax)
            val_mainchunk = self._normalize(val_mainchunk, self.mmax)
            val_meterchunk = self._normalize(val_meterchunk, self.mmax)

            self.train_on_chunk(train_mainchunk, train_meterchunk, val_mainchunk, val_meterchunk, epochs, batch_size)
            try:
                train_mainchunk = next(train_main_power_series)
                train_meterchunk = next(train_meter_power_series)
                val_mainchunk = next(val_main_power_series)
                val_meterchunk = next(val_meter_power_series)
            except:
                run = False

    def train_on_chunk(self, train_mainchunk, train_meterchunk, val_mainchunk, val_meterchunk, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        train_mainchunk : chunk of site meter
        train_meterchunk : chunk of appliance
        epochs : number of epochs for training
        batch_size : size of batch used for training
        '''

        # Replace NaNs with 0s
        train_mainchunk.fillna(0, inplace=True)
        train_meterchunk.fillna(0, inplace=True)
        ix = train_mainchunk.index.intersection(train_meterchunk.index)
        train_mainchunk = np.array(train_mainchunk[ix])
        train_meterchunk = np.array(train_meterchunk[ix])
        train_mainchunk = np.reshape(train_mainchunk, (train_mainchunk.shape[0], 1, 1))

        val_mainchunk.fillna(0, inplace=True)
        val_meterchunk.fillna(0, inplace=True)
        ix = val_mainchunk.index.intersection(val_meterchunk.index)
        val_mainchunk = np.array(val_mainchunk[ix])
        val_meterchunk = np.array(val_meterchunk[ix])
        val_mainchunk = np.reshape(val_mainchunk, (val_mainchunk.shape[0], 1, 1))

        history = self.model.fit(train_mainchunk, train_meterchunk, epochs=epochs, batch_size=batch_size, shuffle=True)
        self.update_logfile(self.train_logfile, history.history['loss'], self.total_epochs)
        self.total_epochs += epochs

        loss = self.model.evaluate(val_mainchunk, val_meterchunk, batch_size=batch_size)
        self.update_logfile(self.val_logfile, [loss], self.total_epochs-1)

    def train_across_buildings(self, train_mainlist, train_meterlist, val_mainlist, val_meterlist, epochs=1,
                               batch_size=128, **load_kwargs):
        '''Train using data from multiple train_buildings

        Parameters
        ----------
        train_mainlist : a list of nilmtk.ElecMeter objects for the aggregate data of each building
        train_meterlist : a list of nilmtk.ElecMeter objects for the meter data of each building
        batch_size : size of batch used for training
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        assert(len(train_mainlist) == len(train_meterlist), "Number of train main and meter channels should be equal")
        assert(len(val_mainlist) == len(val_meterlist), "Number of validation main and meter channels should be equal")
        train_num_meters = len(train_mainlist)
        val_num_meters = len(val_mainlist)

        train_mainps = [None] * train_num_meters
        train_meterps = [None] * train_num_meters
        train_mainchunks = [None] * train_num_meters
        train_meterchunks = [None] * train_num_meters
        val_mainps = [None] * val_num_meters
        val_meterps = [None] * val_num_meters
        val_mainchunks = [None] * val_num_meters
        val_meterchunks = [None] * val_num_meters

        # Get generators of timeseries
        for i,m in enumerate(train_mainlist):
            train_mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(train_meterlist):
            train_meterps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(val_mainlist):
            val_mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(val_meterlist):
            val_meterps[i] = m.power_series(**load_kwargs)

        # Get a chunk of data
        for i in range(train_num_meters):
            train_mainchunks[i] = next(train_mainps[i])
            train_meterchunks[i] = next(train_meterps[i])
        for i in range(val_num_meters):
            val_mainchunks[i] = next(val_mainps[i])
            val_meterchunks[i] = next(val_meterps[i])
        if self.mmax is None:
            self.mmax = max([m.max() for m in train_mainchunks])

        run = True
        while run:
            # Normalize and train
            train_mainchunks = [self._normalize(m, self.mmax) for m in train_mainchunks]
            train_meterchunks = [self._normalize(m, self.mmax) for m in train_meterchunks]
            val_mainchunks = [self._normalize(m, self.mmax) for m in val_mainchunks]
            val_meterchunks = [self._normalize(m, self.mmax) for m in val_meterchunks]

            self.train_across_buildings_chunk(train_mainchunks, train_meterchunks, val_mainchunks, val_meterchunks,
                                              epochs, batch_size)

            # If more chunks, repeat
            try:
                for i in range(train_num_meters):
                    train_mainchunks[i] = next(train_mainps[i])
                    train_meterchunks[i] = next(train_meterps[i])
                for i in range(val_num_meters):
                    val_mainchunks[i] = next(val_mainps[i])
                    val_meterchunks[i] = next(val_meterps[i])
            except:
                run = False

    def train_across_buildings_chunk(self, train_mainchunks, train_meterchunks, val_mainchunks, val_meterchunks, epochs,
                                     batch_size):
        '''Train using only one chunk of data. This chunk consists of data from
        all train_buildings.

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        batch_size : size of batch used for training
        '''
        train_num_meters = len(train_mainchunks)
        train_batch_size = int(batch_size / train_num_meters)
        train_num_of_batches = [None] * train_num_meters

        val_num_meters = len(val_mainchunks)
        val_batch_size = int(batch_size / val_num_meters)
        val_num_of_batches = [None] * val_num_meters

        # Find common parts of timeseries
        for i in range(train_num_meters):
            train_mainchunks[i].fillna(0, inplace=True)
            train_meterchunks[i].fillna(0, inplace=True)
            ix = train_mainchunks[i].index.intersection(train_meterchunks[i].index)
            m1 = train_mainchunks[i]
            m2 = train_meterchunks[i]
            train_mainchunks[i] = m1[ix]
            train_meterchunks[i] = m2[ix]

            train_num_of_batches[i] = int(len(ix) / train_batch_size) - 1

        for i in range(val_num_meters):
            val_mainchunks[i].fillna(0, inplace=True)
            val_meterchunks[i].fillna(0, inplace=True)
            ix = val_mainchunks[i].index.intersection(val_meterchunks[i].index)
            m1 = val_mainchunks[i]
            m2 = val_meterchunks[i]
            val_mainchunks[i] = m1[ix]
            val_meterchunks[i] = m2[ix]

            val_num_of_batches[i] = int(len(ix) / val_batch_size) - 1

        for e in range(epochs): # Iterate for every epoch
            print(e)
            train_loss = 0
            batch_indexes = list(range(min(train_num_of_batches)))
            random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes): # Iterate for every batch
                print("Batch {} of {}".format(bi, min(train_num_of_batches)), end="\n")
                sys.stdout.flush()
                X_batch = np.empty((train_batch_size * train_num_meters, 1, 1))
                Y_batch = np.empty((train_batch_size * train_num_meters, 1))

                # Create a batch out of data from all train_buildings
                for i in range(train_num_meters):
                    mainpart = train_mainchunks[i]
                    meterpart = train_meterchunks[i]
                    mainpart = mainpart[b * train_batch_size:(b + 1) * train_batch_size]
                    meterpart = meterpart[b * train_batch_size:(b + 1) * train_batch_size]
                    X = np.reshape(mainpart, (train_batch_size, 1, 1))
                    Y = np.reshape(meterpart, (train_batch_size, 1))

                    X_batch[i * train_batch_size:(i + 1) * train_batch_size] = np.array(X)
                    Y_batch[i * train_batch_size:(i + 1) * train_batch_size] = np.array(Y)

                # Shuffle data
                p = np.random.permutation(len(X_batch))
                X_batch, Y_batch = X_batch[p], Y_batch[p]

                # Train model
                train_loss += self.model.train_on_batch(X_batch, Y_batch)
            train_loss /= min(train_num_of_batches)
            print("\n")
            self.update_logfile(self.train_logfile, [train_loss], e+self.total_epochs)

            val_loss = 0
            batch_indexes = list(range(min(val_num_of_batches)))
            random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes):  # Iterate for every batch
                print("Batch {} of {}".format(bi, min(val_num_of_batches)), end="\n")
                sys.stdout.flush()
                X_batch = np.empty((val_batch_size * val_num_meters, 1, 1))
                Y_batch = np.empty((val_batch_size * val_num_meters, 1))

                # Create a batch out of data from all train_buildings
                for i in range(val_num_meters):
                    mainpart = val_mainchunks[i]
                    meterpart = val_meterchunks[i]
                    mainpart = mainpart[b * val_batch_size:(b + 1) * val_batch_size]
                    meterpart = meterpart[b * val_batch_size:(b + 1) * val_batch_size]
                    X = np.reshape(mainpart, (val_batch_size, 1, 1))
                    Y = np.reshape(meterpart, (val_batch_size, 1))

                    X_batch[i * val_batch_size:(i + 1) * val_batch_size] = np.array(X)
                    Y_batch[i * val_batch_size:(i + 1) * val_batch_size] = np.array(Y)

                # Shuffle data
                p = np.random.permutation(len(X_batch))
                X_batch, Y_batch = X_batch[p], Y_batch[p]

                # Evaluate model
                val_loss += self.model.test_on_batch(X_batch, Y_batch)
            val_loss /= min(val_num_of_batches)
            print("\n")
            self.update_logfile(self.val_logfile, [val_loss], e+self.total_epochs)

        self.total_epochs += epochs

    def disaggregate(self, mains, output_datastore, results_file, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : a nilmtk.ElecMeter of aggregate data
        meter_metadata: a nilmtk.ElecMeter of the observed meter used for storing the metadata
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())

        # ROTEM
        # bugfix - it was considering meter1 by default.

        mains_data_location = building_path + '/elec/meter%s' % str(mains.instance())
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue

            line = "New sensible chunk: {}".format(len(chunk))
            with open(results_file, "a") as text_file:
                text_file.write(line + '\n')
            print(line)

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series of aggregate data
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        X_batch = np.array(mains)
        X_batch = np.reshape(X_batch, (X_batch.shape[0],1,1))

        pred = self.model.predict(X_batch, batch_size=128)
        pred = np.reshape(pred, (len(pred)))
        column = pd.Series(pred, index=mains.index[:len(X_batch)], name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

    def import_model(self, filename):
        '''Loads keras model from h5

        Parameters
        ----------
        filename : filename for .h5 file

        Returns: Keras model
        '''
        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk

    def _create_model(self):
        '''Creates the RNN module described in the paper
        '''
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(16, 4, activation="linear", input_shape=(1,1), padding="same", strides=1))

        #Bi-directional LSTMs
        model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(Bidirectional(LSTM(256, return_sequences=False, stateful=False), merge_mode='concat'))

        # Fully Connected Layers
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='adam')
        plot_model(model, to_file='model.png', show_shapes=True)

        return model

    @staticmethod
    def init_logfile(logfile):
        with open(logfile, "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['epoch', 'loss'])

    @staticmethod
    def update_logfile(logfile, loss, last_index):
        with open(logfile, "a") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i, j in zip(range(last_index, last_index + len(loss)), loss):
                writer.writerow([i, j])
