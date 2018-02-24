# neural-disaggregator

Implementation of NILM disaggegregator using Neural Networks, using [NILMTK](https://github.com/NILMTK/NILMTK) and Keras.

The architecture is based on [Neural NILM: Deep Neural Networks Applied to Energy Disaggregation](https://arxiv.org/pdf/1507.06594.pdf) by Jack Kelly and William Knottenbelt.

The implemented model is a Recurrent network with LSTM neurons as mentioned in [Neural NILM](https://arxiv.org/pdf/1507.06594.pdf),
extended by adding units to the convolutional and recurrent layers and by using dropout.

To start training use ukdale-test.py or ukdale-across-buildings-test.py and provide the parameters:
- Select the time frames as well as buildings for the training, validation and test sets.
- Select the sample_period (in secs), meter_key (= target appliance) and learning_rate.
