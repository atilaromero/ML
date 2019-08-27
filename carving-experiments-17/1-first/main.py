import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Activation, TimeDistributed, Flatten

sys.path.append('..')
from run_experiments import Experiment, run_experiments, save_experiment_results

def D():
    last = l0 = Input(shape=(512,256))
    last = Flatten()(last)
    last = Dense(4)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'first')

def LD():
    last = l0 = Input(shape=(512,256))
    last = LSTM(32)(last)
    last = Dense(4)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'first')

def CL():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(3, (32,), strides=32)(last)
    last = LSTM(4)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'first')

experiments = [
    D(),
    LD(),
    CL(),
]

results = []
for d in run_experiments(experiments,
        batch_size=100,
        validation_batch_size=10,
        validation_steps=10,
        steps_per_epoch=10,
        epochs=150,
        val_acc_limit=0.9):
    print(d)
    results.append(d)

save_experiment_results('experiments.tsv', results)
