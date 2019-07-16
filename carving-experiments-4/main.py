import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Activation, TimeDistributed, Flatten

sys.path.append('..')
from run_experiments import Experiment, run_experiments, save_experiment_results

NINPUT=1024
NOUPUT=1

# 12
def D():
    last = l0 = Input(shape=(NINPUT,256))
    last = Flatten()(last)
    last = Dense(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')

# ?
def LD():
    last = l0 = Input(shape=(NINPUT,256))
    last = LSTM(32)(last)
    last = Dense(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')


#15, 14, 17
def CL():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(3, (32,), strides=32)(last)
    last = LSTM(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')

def CCL():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(128, (8,), strides=8)(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = LSTM(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
def CCLL():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(128, (8,), strides=8)(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 18
def CMCML():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(128, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = LSTM(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 20
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


# 13
def CLL():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(32, (32,), strides=32)(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 16
def CML():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(32, (32,), strides=32)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)    
    last = LSTM(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 23
def CLD():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(256, (16,), strides=16)(last)
    last = LSTM(128)(last)
    last = Dense(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')


# 25
def CD():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(64,(64,),strides=8)(last)
    last = Flatten()(last)
    last = Dense(NOUPUT)(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 26
def CM():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(NOUPUT, (32,), strides=1)(last)
    last = MaxPooling1D(pool_size=481, strides=1)(last)
    last = Flatten()(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 27
def CCM():
    last = l0 = Input(shape=(NINPUT,256))
    last = Conv1D(3, (32,), strides=1)(last)
    last = Conv1D(NOUPUT, (2,), strides=2)(last)
    last = MaxPooling1D(pool_size=240, strides=1)(last)
    last = Flatten()(last)
    last = Activation('sigmoid')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')

experiments = [
    D(),
    LD(),
    CL(),
    CCL(),
    CCLL(),
    CMCML(),
    CMCMLL(),
    CLL(),
    CML(),
    CLD(),
    CD(),
    CM(),
    CCM(),
]

results = []
for d in run_experiments(experiments,
        batch_size=10,
        validation_batch_size=10,
        validation_steps=100,
        steps_per_epoch=100,
        epochs=600):
    print(d)
    results.append(d)

save_experiment_results('experiments.tsv', results)

