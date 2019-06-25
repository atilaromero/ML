import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Activation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../..')
import utils
import utils.load
import utils.sampler

def get_model():
    last = l0 = Input(shape=(512,256))
    last = Conv1D(128, (3,), padding="same", activation="relu")(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(64, (3,), padding="same", activation="relu")(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(32, (3,), padding="same", activation="relu")(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(16, (3,), padding="same", activation="relu")(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = Conv1D(8, (3,), padding="same", activation="relu")(last)
    last = MaxPooling1D(pool_size=2, strides=2, data_format='channels_first')(last)
    last = LSTM(16, return_sequences=True)(last)
    last = LSTM(16, return_sequences=False)(last)
    last = Dense(3)(last)
    last = Activation('softmax')(last)

    model = tf.keras.Model([l0], last)
    model.summary()
    return model

def compile(model):
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'])

categories = ['pdf','png', 'jpg']
ix_to_cat = dict([(i,x) for i,x in enumerate(categories)])
cat_to_ix = dict([(x,i) for i,x in enumerate(categories)])

def sample_sector(path):
    with open(path, 'rb') as f:
        f.seek(0,2)
        size = f.tell()
        i = np.random.randint(0,size)
        sector = i//512
        f.seek(sector*512,0)
        b = f.read(512)
        n = np.zeros((512),dtype='int')
        n[:len(b)] = [int(x) for x in b]
        return n

def ys_from_filenames(filenames):
    cats = [utils.load.category_from_extension(s) for s in filenames]
    cats = [cat_to_ix[x] for x in cats]
    ys = utils.one_hot(cats, len(categories))
    return ys

def xs_from_filenames(filenames):
    xs = np.zeros((len(filenames),512,256))
    for i,f in enumerate(filenames):
        x = sample_sector(f)
        xs[i] = utils.one_hot(x,256)
    return xs

def sector_generator(filenames):
    while True:
        xs = xs_from_filenames(filenames)
        ys = ys_from_filenames(filenames)
        yield xs, ys

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_file):
        self.save_file = save_file
    def on_epoch_end(self, epoch, logs):
        if epoch % 5 == 0:
            self.model.save(self.save_file)
            if logs['acc'] > 0.9:
                self.model.stop_training = True

def train(model, save_file, examplesFolder):
    examples = utils.load.examples_from(examplesFolder)
    history = model.fit_generator(sector_generator(examples),
        steps_per_epoch=10,
        epochs=1000,
        callbacks=[MyCallback(save_file)],
    )

def evaluate(model, examplesFolder, sample_size=1000):
    examples = utils.load.examples_from(examplesFolder)
    sample = utils.sampler.choice(examples, sample_size)
    ys = ys_from_filenames(sample)
    xs = xs_from_filenames(sample)
    model.evaluate(xs, ys)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        model = get_model()
        utils.load.maybe_load_weigths(model, 'model.h5')
        compile(model)
        train(model, 'model.h5', '../../datasets/carving/3files')
        exit(0)
    if sys.argv[1] == 'train':
        model = get_model()
        utils.load.maybe_load_weigths(model, sys.argv[2])
        compile(model)
        train(model, *sys.argv[2:])
    if sys.argv[1] == 'evaluate':
        model = get_model()
        compile(model)
        utils.load.maybe_load_weigths(model, sys.argv[2])
        evaluate(model, *sys.argv[3:])
    else:
        print(f'Use {sys.argv[0]} COMMAND')
        print(f'commands:')
        print(f'    train save_file examplesFolder sample_size epochs batch_size')
        print(f'    evaluate save_file examplesFolder sample_size')
        exit(1)
