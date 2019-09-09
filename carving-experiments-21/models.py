import sys
import itertools
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy, mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2

def compile(model, loss):
    model.compile(loss=loss,
        optimizer=Adam(),
        metrics=[categorical_accuracy, binary_accuracy])



def C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D31_cat(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(64, (16,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(64, (32,), strides=2, padding='same', activation='relu')(last)
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C256_16_16_L128_D31_cat(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(256, (16,), strides=16)(last)
    last = LSTM(128)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C64_16_2pr_5C32_4_2pr_C64_32_2pr_F_D31_cat(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(64, (16,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(64, (32,), strides=2, padding='same', activation='relu')(last)
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C64_16_1pr_5C32_4_1pr_C64_32_1pr_L64_D31_cat(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(64, (16,), strides=1, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
    last = Conv1D(64, (32,), strides=1, padding='same', activation='relu')(last)
    last = LSTM(64)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

def C32_4_2PR_C64_32_2PR_F_D31_cat(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(64, (32,), strides=2, padding='same', activation='relu')(last)
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model([l0], last, name=myfuncname)
    compile(model, loss)
    return model

# def C64_64_8pr_C64_16_16pr_L64_D31_cat(classes, len_byte_vector, activation, loss):
#     myfuncname = sys._getframe().f_code.co_name
#     last = l0 = Input(shape=(512,len_byte_vector))
#     last = Conv1D(64, (16,), strides=2, padding='same', activation='relu')(last)
#     last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
#     last = LSTM(64)(last)
#     last = Dense(classes)(last)
#     last = Activation(activation)(last)
#     model = Model([l0], last, name=myfuncname)
#     compile(model, loss)
#     return model



def args_gen(*args, **kwargs):
    if args == []:
        args = [None]
    if len(kwargs.keys()) == 0:
        kwargs['dummy'] = [None]
    for _args in itertools.product(*args):
        keys, values = list(zip(*kwargs.items()))
        for _values in itertools.product(*values):
            _kwargs = dict(zip(keys, _values))
            # remove None values
            __kwargs = dict([(k,v) for k,v in _kwargs.items() if v != None])
            __args = [x for x in _args if x != None]
            yield __args,__kwargs

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

def activation_from(loss):
    if loss[:3] in ['cat']:
        return Activation('softmax')
    if loss[:3] in ['bin', 'mse']:
        return Activation('sigmoid')
    raise Exception('activation_from need to be updated to acept loss =', loss)

        
def name_from(layer, *args, **kwargs):
    if layer == activation_from:
        return args[0][:3]
    if issubclass(layer, LSTM):
        if 'return_sequences' in kwargs:
            return 'LV%d'%args[0]
        return 'L%d'%args[0]
    if issubclass(layer, Conv1D):
        nc = 'C%d_%d_%d'%(args[0],args[1][0],kwargs['strides'])
        if 'padding' in kwargs:
            nc += 'p'
        if 'activation' in kwargs:
            nc += 'r'
        if 'kernel_regularizer' in kwargs:
            nc += 'k'
        return nc
    if issubclass(layer, Dense):
        return 'D%d'%args[0]
    if issubclass(layer, Flatten):
        return 'F'
    if issubclass(layer, MaxPooling1D):
        return 'M'
    if issubclass(layer, BatchNormalization):
        return 'B'
    if issubclass(layer, Activation):
        return 'A'
    raise Exception('name_from does not know', layer)

def genall(classes, shape):
    def endLD(units):
        return [
            [(LSTM, x, y) for x,y in args_gen(
                [units],
            )],
            [(Dense, x, y) for x,y in args_gen(
                [classes],
            )],
        ]
    endFD = [
        [(Flatten, x, y) for x,y in args_gen()],
        [(Dense, x, y) for x,y in args_gen(
            [classes],
        )],
    ]
    endL = [
        [(LSTM, x, y) for x,y in args_gen(
            [classes],
        )],
    ]
    def endCM(pool_size):
        return [
            [(Conv1D, x, y) for x,y in args_gen(
                [classes], 
                [(1,)], 
                strides=[1], 
            )],
            [(MaxPooling1D, x, y) for x,y in args_gen(
                pool_size=[pool_size],
                strides=[1],
            )],
            [(Flatten, x, y) for x,y in args_gen()],
        ]
    layers = [
        [(Input, x, y) for x,y in args_gen(shape=[shape])],

        [(Conv1D, x, y) for x,y in args_gen(
            [64], 
            [(16,)], 
            strides=[2], 
            padding=['same'], 
            # activation=['relu'],
            # kernel_regularizer=[l2(0.01)],
        )],
        [(BatchNormalization, x, y) for x,y in args_gen()],
        [(Activation, x, y) for x,y in args_gen(['relu'])],
        [(Conv1D, x, y) for x,y in args_gen(
            [32], 
            [(4,)], 
            strides=[2], 
            padding=['same'], 
            activation=['relu'],
            # kernel_regularizer=[l2(0.01)],
        )],
        [(BatchNormalization, x, y) for x,y in args_gen()],
        [(Activation, x, y) for x,y in args_gen(['relu'])],
        [(Conv1D, x, y) for x,y in args_gen(
            [64], 
            [(32,)], 
            strides=[2], 
            padding=['same'], 
            activation=['relu'],
            # kernel_regularizer=[l2(0.01)],
        )],
        [(BatchNormalization, x, y) for x,y in args_gen()],
        [(Activation, x, y) for x,y in args_gen(['relu'])],
        # [(LSTM, x, y) for x,y in args_gen(
        #     [64],
        #     return_sequences=[True],
        # )],
        *endFD,
        # *endLD(32),
        # *endL,
        # *endCM(8),
        [(activation_from, x, y) for x,y in args_gen([
            'categorical_crossentropy', 
            # 'binary_crossentropy', 
            # 'mse',
            ]
        )],
    ]
    for m in itertools.product(*layers):
        l0_spec = m[0]
        last = l0 = l0_spec[0](*l0_spec[1], **l0_spec[2])
        name = []
        for l in m[1:]:
            last = l[0](*l[1], **l[2])(last)
            name.append(name_from(l[0],*l[1], **l[2]))
        name = '_'.join(name)
        print(name)
        model = Model([l0], last, name=name)
        loss = m[-1][1][0]
        compile(model, loss)
        yield model

if __name__ == '__main__':
    for m in genall(31, shape=(512,256)):
        m.summary()