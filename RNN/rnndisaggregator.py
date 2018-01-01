from __future__ import print_function, division

import csv
import random
import sys
import h5py

import pandas as pd
import numpy as np

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Bidirectional, Dropout, TimeDistributed
from keras.utils import plot_model
from nilmtk.disaggregate import Disaggregator


SEQUENCE_LENGTH = 128


class RNNDisaggregator(Disaggregator):
    """
    Attempt to create a RNN Disaggregator.

    Attributes
    ----------
    model : Keras Sequential model.
    mmax : The maximum value of the aggregate data.
    MIN_CHUNK_LENGTH : int
        The minimum length of an acceptable chunk.
    """

    def __init__(self, train_logfile, val_logfile):
        """
        Initialize disaggregator.

        :param train_logfile: Training loss loggin.
        :param val_logfile: Validation loss loggin.
        """

        self.MODEL_NAME = "LSTM"
        self.mmax = None
        self.std = None
        self.MIN_CHUNK_LENGTH = 100
        self.model = self._create_model()
        self.total_epochs = 0  # track total training epochs
        self.train_logfile = train_logfile
        self.val_logfile = val_logfile

        self.init_logfile(train_logfile)
        self.init_logfile(val_logfile)

    def train(self, train_mains, train_meter, validation_mains, validation_meter, epochs=1, batch_size=16, **load_kwargs):
        """
        Train model.

        :param train_mains: nilmtk.ElecMeter object for the training aggregate data.
        :param train_meter: nilmtk.ElecMeter object for the training meter data.
        :param validation_mains: nilmtk.ElecMeter object for the validation aggregate data.
        :param validation_meter: nilmtk.ElecMeter object for the validation meter data.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of batch used for training.
        :param load_kwargs: Keyword arguments passed to train_meter.power_series()
        """

        train_main_power_series = train_mains.power_series(**load_kwargs)
        train_meter_power_series = train_meter.power_series(**load_kwargs)
        val_main_power_series = validation_mains.power_series(**load_kwargs)
        val_meter_power_series = validation_meter.power_series(**load_kwargs)

        # train chunks
        run = True
        train_mainchunk = next(train_main_power_series)
        train_meterchunk = next(train_meter_power_series)
        val_mainchunk = next(val_main_power_series)
        val_meterchunk = next(val_meter_power_series)
        if self.mmax is None:
            self.mmax = train_meterchunk.max()

        while run:
            self.train_on_chunk(train_mainchunk, train_meterchunk, val_mainchunk, val_meterchunk, epochs, batch_size)
            try:
                train_mainchunk = next(train_main_power_series)
                train_meterchunk = next(train_meter_power_series)
                val_mainchunk = next(val_main_power_series)
                val_meterchunk = next(val_meter_power_series)
            except:
                run = False

    def train_on_chunk(self, train_mainchunk, train_meterchunk, val_mainchunk, val_meterchunk, epochs, batch_size):
        """
        Train using only one chunk.

        :param train_mainchunk: Training chunk of site meter.
        :param train_meterchunk: Training chunk of appliance.
        :param val_mainchunk: Validation chunk of site meter.
        :param val_meterchunk: Validation chunk of appliance.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of batch used for training.
        """

        """
        Series lengths (for sample_period = 6):
        --------------
        kettle => 128
        dish washer => 1536
        fridge => 512
        """

        # replace NaNs with 0s
        train_mainchunk.fillna(0, inplace=True)
        train_meterchunk.fillna(0, inplace=True)
        ix = train_mainchunk.index.intersection(train_meterchunk.index)
        train_mainchunk = np.array(train_mainchunk[ix])
        train_meterchunk = np.array(train_meterchunk[ix])

        # shape should be determined according to the target appliance
        # truncate dataset if necessary
        if train_mainchunk.shape[0] % SEQUENCE_LENGTH != 0:
            length = int(train_mainchunk.shape[0] / SEQUENCE_LENGTH) * SEQUENCE_LENGTH
            train_mainchunk = train_mainchunk[:length]
            train_meterchunk = train_meterchunk[:length]
        train_mainchunk = np.reshape(train_mainchunk, (-1, SEQUENCE_LENGTH, 1))
        train_meterchunk = np.reshape(train_meterchunk, (-1, SEQUENCE_LENGTH, 1))

        if self.std is None:
            rand_idx = random.randint(0, train_mainchunk.shape[0]-1)
            self.std = train_mainchunk[rand_idx].std()
            while self.std == 0:
                rand_idx = random.randint(0, train_mainchunk.shape[0]-1)
                self.std = train_mainchunk[rand_idx].std()

        train_mainchunk = self._normalize(train_mainchunk)
        train_meterchunk = self._normalize_targets(train_meterchunk)

        # replace NaNs with 0s
        val_mainchunk.fillna(0, inplace=True)
        val_meterchunk.fillna(0, inplace=True)
        ix = val_mainchunk.index.intersection(val_meterchunk.index)
        val_mainchunk = np.array(val_mainchunk[ix])
        val_meterchunk = np.array(val_meterchunk[ix])

        # truncate dataset if necessary
        if val_mainchunk.shape[0] % SEQUENCE_LENGTH != 0:
            length = int(val_mainchunk.shape[0] / SEQUENCE_LENGTH) * SEQUENCE_LENGTH
            val_mainchunk = val_mainchunk[:length]
            val_meterchunk = val_meterchunk[:length]
        val_mainchunk = np.reshape(val_mainchunk, (-1, SEQUENCE_LENGTH, 1))
        val_meterchunk = np.reshape(val_meterchunk, (-1, SEQUENCE_LENGTH, 1))

        history = self.model.fit(train_mainchunk, train_meterchunk, epochs=epochs, batch_size=batch_size, shuffle=True)
        self.update_logfile(self.train_logfile, history.history['loss'], self.total_epochs)
        self.total_epochs += epochs

        loss = self.model.evaluate(val_mainchunk, val_meterchunk, batch_size=batch_size)
        self.update_logfile(self.val_logfile, [loss], self.total_epochs-1)

    def train_across_buildings(self, train_mainlist, train_meterlist, val_mainlist, val_meterlist, epochs=1,
                               batch_size=16, **load_kwargs):
        """
        Train using data from multiple buildings.

        :param train_mainlist: List of nilmtk.ElecMeter objects for the training aggregate data of each building.
        :param train_meterlist: List of nilmtk.ElecMeter objects for the training meter data of each building.
        :param val_mainlist: List of nilmtk.ElecMeter objects for the validation aggregate data of each building.
        :param val_meterlist: List of nilmtk.ElecMeter objects for the validation meter data of each building.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of batch used for training.
        :param load_kwargs: Keyword arguments passed to meter.power_series()
        """

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

        # get generators of timeseries
        for i,m in enumerate(train_mainlist):
            train_mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(train_meterlist):
            train_meterps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(val_mainlist):
            val_mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(val_meterlist):
            val_meterps[i] = m.power_series(**load_kwargs)

        # get a chunk of data
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
            # normalize and train
            train_mainchunks = [self._normalize(m, self.mmax) for m in train_mainchunks]
            train_meterchunks = [self._normalize(m, self.mmax) for m in train_meterchunks]
            val_mainchunks = [self._normalize(m, self.mmax) for m in val_mainchunks]
            val_meterchunks = [self._normalize(m, self.mmax) for m in val_meterchunks]

            self.train_across_buildings_chunk(train_mainchunks, train_meterchunks, val_mainchunks, val_meterchunks,
                                              epochs, batch_size)

            # if more chunks, repeat
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
        """
        Train using only one chunk of data. This chunk consists of data from all buildings.
        :param train_mainchunks: Training chunk of site meter.
        :param train_meterchunks: Training chunk of appliance.
        :param val_mainchunks: Validation chunk of site meter.
        :param val_meterchunks: Validation chunk of appliance.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of batch used for training.
        """

        train_num_meters = len(train_mainchunks)
        train_batch_size = int(batch_size / train_num_meters)
        train_num_of_batches = [None] * train_num_meters

        val_num_meters = len(val_mainchunks)
        val_batch_size = int(batch_size / val_num_meters)
        val_num_of_batches = [None] * val_num_meters

        # find common parts of timeseries
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

        for e in range(epochs):  # iterate for every epoch
            print(e)
            train_loss = 0
            batch_indexes = list(range(min(train_num_of_batches)))
            random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes):  # iterate for every batch
                print("Batch {} of {}".format(bi, min(train_num_of_batches)), end="\n")
                sys.stdout.flush()
                X_batch = np.empty((train_batch_size * train_num_meters, 1, 1))
                Y_batch = np.empty((train_batch_size * train_num_meters, 1))

                # create a batch out of data from all buildings
                for i in range(train_num_meters):
                    mainpart = train_mainchunks[i]
                    meterpart = train_meterchunks[i]
                    mainpart = mainpart[b * train_batch_size:(b + 1) * train_batch_size]
                    meterpart = meterpart[b * train_batch_size:(b + 1) * train_batch_size]
                    X = np.reshape(mainpart, (train_batch_size, 1, 1))
                    Y = np.reshape(meterpart, (train_batch_size, 1))

                    X_batch[i * train_batch_size:(i + 1) * train_batch_size] = np.array(X)
                    Y_batch[i * train_batch_size:(i + 1) * train_batch_size] = np.array(Y)

                # shuffle data
                p = np.random.permutation(len(X_batch))
                X_batch, Y_batch = X_batch[p], Y_batch[p]

                # train model
                train_loss += self.model.train_on_batch(X_batch, Y_batch)
            train_loss /= min(train_num_of_batches)
            print("\n")
            self.update_logfile(self.train_logfile, [train_loss], e+self.total_epochs)

            val_loss = 0
            batch_indexes = list(range(min(val_num_of_batches)))
            random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes):  # iterate for every batch
                print("Batch {} of {}".format(bi, min(val_num_of_batches)), end="\n")
                sys.stdout.flush()
                X_batch = np.empty((val_batch_size * val_num_meters, 1, 1))
                Y_batch = np.empty((val_batch_size * val_num_meters, 1))

                # create a batch out of data from all buildings
                for i in range(val_num_meters):
                    mainpart = val_mainchunks[i]
                    meterpart = val_meterchunks[i]
                    mainpart = mainpart[b * val_batch_size:(b + 1) * val_batch_size]
                    meterpart = meterpart[b * val_batch_size:(b + 1) * val_batch_size]
                    X = np.reshape(mainpart, (val_batch_size, 1, 1))
                    Y = np.reshape(meterpart, (val_batch_size, 1))

                    X_batch[i * val_batch_size:(i + 1) * val_batch_size] = np.array(X)
                    Y_batch[i * val_batch_size:(i + 1) * val_batch_size] = np.array(Y)

                # shuffle data
                p = np.random.permutation(len(X_batch))
                X_batch, Y_batch = X_batch[p], Y_batch[p]

                # evaluate model
                val_loss += self.model.test_on_batch(X_batch, Y_batch)
            val_loss /= min(val_num_of_batches)
            print("\n")
            self.update_logfile(self.val_logfile, [val_loss], e+self.total_epochs)

        self.total_epochs += epochs

    def disaggregate(self, mains, output_datastore, results_file, meter_metadata, **load_kwargs):
        """
        Disaggregate mains according to the model learnt previously.

        :param mains: nilmtk.ElecMeter of aggregate data.
        :param output_datastore: Instance of nilmtk.DataStore subclass for storing power predictions from disaggregation
            algorithm.
        :param results_file: Output text file.
        :param meter_metadata: nilmtk.ElecMeter of the observed meter used for storing the metadata.
        :param load_kwargs: Keyword arguments passed to mains.power_series(**kwargs)
        """

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())

        # ROTEM
        # bugfix - it was considering meter1 by default.

        main_meter = mains.instance()
        if isinstance(main_meter, tuple):  # if a tuple pick the first item (for printing purposes only)
            main_meter = main_meter[0]

        mains_data_location = building_path + '/elec/meter%s' % str(main_meter)
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

            appliance_power = self.disaggregate_chunk(chunk)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize_targets(appliance_power)

            # append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # save metadata to output
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
        """
        In-memory disaggregation.

        :param mains: pd.Series of aggregate data.
        :return: appliance powers - pd.DataFrame where each column represents a disaggregated appliance. Column names
            are the integer index into self.model for the appliance in question.
        """

        mains.fillna(0, inplace=True)
        X_batch = np.array(mains)
        if X_batch.shape[0] % SEQUENCE_LENGTH != 0:
            length = int(X_batch.shape[0] / SEQUENCE_LENGTH) * SEQUENCE_LENGTH
            X_batch = X_batch[:length]
        X_batch = np.reshape(X_batch, (-1, SEQUENCE_LENGTH, 1))
        X_batch = self._normalize(X_batch)

        pred = self.model.predict(X_batch, batch_size=16)
        pred = np.reshape(pred, (pred.shape[0]*SEQUENCE_LENGTH))
        column = pd.Series(pred, index=mains.index[:(X_batch.shape[0]*SEQUENCE_LENGTH)], name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

    # TODO: update
    def import_model(self, filename):
        """
        Loads keras model from h5.

        :param filename: Filename for .h5 file.
        :return: Keras model.
        """

        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]

    def export_model(self, filename):
        """
        Saves keras model to h5.

        :param filename: Filename for .h5 file.
        """

        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])

    def _normalize(self, chunk):
        """
        Normalizes timeseries. Each sequence is normalized to have zero mean and then divided by the std of a random
        sample of the training set.

        :param chunk: The timeseries to normalize.
        :return: Normalized timeseries.
        """

        mean = chunk.mean(axis=1)
        mean = np.reshape(mean, (-1,1,1))
        chunk -= mean
        chunk /= self.std
        return chunk

    def _normalize_targets(self, chunk):
        """
        Normalizes the target power demand into the range [0,1].

        :param chunk: The timeseries to normalize.
        :return: Normalized timeseries.
        """

        tchunk = chunk / self.mmax
        return tchunk

    def _denormalize_targets(self, chunk):
        """
        Denormalizes the target power demand into their original range.
        Note: This is not entirely correct.

        :param chunk: The timeseries to denormalize.
        :return: Denormalized timeseries.
        """

        tchunk = chunk * self.mmax
        return tchunk

    @staticmethod
    def _create_model():
        """
        Creates the RNN module described in the paper.
        """

        model = Sequential()

        # 1D Conv
        model.add(Conv1D(16, 4, activation="linear", input_shape=(None,1), padding="same", strides=1))

        # Bi-directional LSTMs
        model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(Bidirectional(LSTM(256, return_sequences=True, stateful=False), merge_mode='concat'))

        # Fully Connected Layers
        model.add(TimeDistributed(Dense(128, activation='tanh')))
        model.add(TimeDistributed(Dense(1, activation='linear')))

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
