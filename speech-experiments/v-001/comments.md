Experiment v-001 uses a model with three convolutional layers of 16, 8 and 4 units each, followed by a LSTM layer of 64 units and a fully connected layer of 27 output units, with softmax activation.

The code is configured to run for 1000 epochs, with only 5 samples each, one for each vowel, taking 1m08s to run in the hardware used.