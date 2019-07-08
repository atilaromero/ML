import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Activation, TimeDistributed, Flatten

sys.path.append('..')
from run_experiments import Experiment, run_experiments, save_experiment_results

# 12
def D():
    last = l0 = Input(shape=(512,256))
    last = Flatten()(last)
    last = Dense(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')

# ?
def LD():
    last = l0 = Input(shape=(512,256))
    last = LSTM(32)(last)
    last = Dense(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')


#15, 14, 17
def CL():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(3, (32,), strides=32)(last)
    last = LSTM(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# def X15():
#     last = l0 = Input(shape=(512,256))
#     last = Conv1D(32, (32,), strides=32)(last)
#     last = LSTM(3)(last)
#     last = Activation('softmax')(last)
#     model = tf.keras.Model([l0], last)
#     myfuncname = sys._getframe().f_code.co_name
#     return Experiment(myfuncname, model, 'all')
# def X14():
#     last = l0 = Input(shape=(512,256))
#     last = Conv1D(3, (32,), strides=16)(last)
#     last = LSTM(3)(last)
#     last = Activation('softmax')(last)
#     model = tf.keras.Model([l0], last)
#     myfuncname = sys._getframe().f_code.co_name
#     return Experiment(myfuncname, model, 'all')
# def X17():
#     last = l0 = Input(shape=(512,256))
#     model = tf.keras.Model([l0], last)
#     last = Conv1D(256, (16,), strides=16)(last)
#     last = LSTM(3)(last)
#     last = Activation('softmax')(last)
#     myfuncname = sys._getframe().f_code.co_name
#     return Experiment(myfuncname, model, 'all')

def CCL():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(128, (8,), strides=8)(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = LSTM(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
def CCLL():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(128, (8,), strides=8)(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 18
def CMCML():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(128, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = LSTM(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 20
def CMCMLL():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(128, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')


# 13
def CLL():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(32, (32,), strides=32)(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 16
def CML():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(32, (32,), strides=32)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)    
    last = LSTM(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 23
def CLD():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(256, (16,), strides=16)(last)
    last = LSTM(128)(last)
    last = Dense(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')


# 25
def CD():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(64,(64,),strides=8)(last)
    last = Flatten()(last)
    last = Dense(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 26
def CM():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(3, (32,), strides=1)(last)
    last = MaxPooling1D(pool_size=481, strides=1)(last)
    last = Flatten()(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
# 27
def CCM():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(3, (32,), strides=1)(last)
    last = Conv1D(3, (2,), strides=2)(last)
    last = MaxPooling1D(pool_size=240, strides=1)(last)
    last = Flatten()(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')

experiments = [
    # D(),
    # LD(),
    # CL(),
    # CCL(),
    # CCLL(),
    # CMCML(),
    # CMCMLL(),
    CLL(),
    CML(),
    CLD(),
    # CD(),
    # CM(),
    # CCM(),
]

results = []
for d in run_experiments(experiments,
        batch_size=10,
        validation_batch_size=10,
        validation_steps=100,
        steps_per_epoch=100,
        epochs=150):
    print(d)
    results.append(d)

save_experiment_results('experiments.tsv', results)

#############################
def X24():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(256,(2,),strides=2)(last)
    last = Conv1D(128,(2,),strides=2)(last)
    last = Conv1D(64,(2,),strides=2)(last)
    last = Conv1D(32,(2,),strides=2)(last)
    last = LSTM(32)(last)
    last = Dense(3)(last)
    last = Activation('softmax')(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model, 'all')
