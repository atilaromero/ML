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


def L32D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = LSTM(32)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def L32L(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = LSTM(32, return_sequences=True)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model


def Cn_32_32L(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(classes, (32,), strides=32)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C128_8_8C64_8_8L(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(128, (8,), strides=8)(last)
    last = Conv1D(64, (8,), strides=8)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C128_8_8C64_8_8L64L(classes, len_byte_vector, activation, loss):
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

def C128_8_8M2C64_8_8M2L(classes, len_byte_vector, activation, loss):
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

def C128_8_8M2C64_8_8M2L64L(classes, len_byte_vector, activation, loss):
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

def C32_32_32L64L(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(32, (32,), strides=32)(last)
    last = LSTM(64, return_sequences=True)(last)
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C32_32_32M2L(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(32, (32,), strides=32)(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)    
    last = LSTM(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C256_16_16L64D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(256, (16,), strides=16)(last)
    last = LSTM(128)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C256_16_16C256_16_1PL128D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(256, (16,), strides=16)(last)
    last = Conv1D(256, (16,), strides=1, padding='same')(last)
    last = LSTM(128)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C256_16_16RC256_16_1PRL128D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(256, (16,), strides=16, activation='relu')(last)
    last = Conv1D(256, (16,), strides=1, padding='same', activation='relu')(last)
    last = LSTM(128)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C256_16_16RL128D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(256, (16,), strides=16, activation='relu')(last)
    last = LSTM(128)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C64_64_8D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(64,(64,),strides=8)(last)
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C32_1M(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(classes, (32,), strides=1)(last)
    last = MaxPooling1D(pool_size=481, strides=1)(last)
    last = Flatten()(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C32_1C2_2M(classes, len_byte_vector, activation, loss):
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

def C_gen(last):
    for u in [256,]:
        for w in [16,]:
            for s in [16,]:
                if s>w:
                    continue
                for p in [False, True]:
                    for r in [False, True]:
                        options = {}
                        name ='C%d_%d_%d'%(u,w,s)
                        if p:
                            options['padding'] = 'same'
                            name += 'P'
                        if r:
                            options['activation'] = 'relu'
                            name += 'R'
                        last = Conv1D(u, (w,), strides=s, **options)(last)
                        yield last, name

def L_gen(last):
    for u in [64]:
        yield LSTM(u)(last), 'L%d'%u

def pre_last_gen(classes, l0):
    if len(l0.shape) == 2:
        yield Dense(classes)(l0), 'D'
    elif len(l0.shape) == 3:
        yield LSTM(classes)(l0), 'L'
        last = Flatten()(l0)
        yield Dense(classes)(last), 'D'
        last = l0
        name = ''
        pool_size = int(l0.shape[-2])
        if l0.shape[-1]!=classes:
            last = Conv1D(classes, (1,), strides=1)(l0)
            name = 'C32_1_1'
        last = MaxPooling1D(pool_size=pool_size, strides=1)(last)
        last = Flatten()(last)
        yield last, name+'M'

def last_gen(l0):
    for activation, loss, name in [
            ('softmax', 'categorical_crossentropy', '_cat'),
            ('sigmoid', 'binary_crossentropy', '_bin'),
            ('sigmoid', 'mse', '_mse'),
        ]:
        last = Activation(activation)(l0)
        yield last, name, loss

def genall(classes, l0):
    for c, nc in C_gen(l0):
        for ll, nll in L_gen(c):
            for pl, npl in pre_last_gen(classes, ll):
                for lt, nlt, loss in last_gen(pl):
                    name = nc+nll+npl+nlt
                    model = Model([l0], lt, name=name)
                    compile(model, loss)
                    yield model

if __name__ == '__main__':
    for m in genall(31, Input(shape=(512,256))):
        print(m.name)