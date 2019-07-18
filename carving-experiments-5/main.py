import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Activation, TimeDistributed, Flatten

sys.path.append('..')
from run_experiments import Experiment, run_experiments, save_experiment_results

NINPUT=1024
NOUPUT=1

def CMCMLL():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(128, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')

experiments = [
    CMCMLL(),
]

results = []
for d in run_experiments(experiments,
        batch_size=100,
        validation_batch_size=100,
        validation_steps=100,
        steps_per_epoch=100,
        epochs=10000):
    print(d)
    results.append(d)

save_experiment_results('experiments.tsv', results)

