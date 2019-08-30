import sys
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Activation
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy, mean_squared_error
from tensorflow.keras.optimizers import Adam

def compile(model, loss):
    model.compile(loss=loss,
        optimizer=Adam(),
        metrics=[categorical_accuracy, binary_accuracy])

def D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model


def LD(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = LSTM(32)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def LL(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = LSTM(32, return_sequences=True)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model


def CL(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(classes, (32,), strides=32)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CCL(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(128, (8,), strides=8)(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CCLL(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(128, (8,), strides=8)(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CMCML(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(128, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CMCMLL(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(128, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CLL(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(32, (32,), strides=32)(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CML(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(32, (32,), strides=32)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)    
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CLD(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(256, (16,), strides=16)(last)
    last = LSTM(128)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CD(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(64,(64,),strides=8)(last)
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CM(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(classes, (32,), strides=1)(last)
    last = MaxPooling1D(pool_size=481, strides=1)(last)
    last = Flatten()(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def CCM(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(classes, (32,), strides=1)(last)
    last = Conv1D(classes, (2,), strides=2)(last)
    last = MaxPooling1D(pool_size=240, strides=1)(last)
    last = Flatten()(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model
