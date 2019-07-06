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
    last = l0 = tf.keras.layers.Input(shape=(None,256))
    last = tf.keras.layers.SimpleRNN(128, return_sequences=True, dropout=0.1)(last)
    last = tf.keras.layers.SimpleRNN(64, return_sequences=True, dropout=0.1)(last)
    last = tf.keras.layers.SimpleRNN(32, return_sequences=True, dropout=0.1)(last)
    last = tf.keras.layers.SimpleRNN(16, return_sequences=True, dropout=0.1)(last)
    last = tf.keras.layers.LSTM(8, return_sequences=False, dropout=0.1)(last)
    last = tf.keras.layers.Dense(3)(last)
    last = tf.keras.layers.Activation('softmax')(last)

    model = tf.keras.Model([l0], last)
    model.summary()
    return model

def compile(model):
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
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

def first_sector(path):
    with open(path, 'rb') as f:
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
        x = first_sector(f)
        xs[i] = utils.one_hot(x,256)
    return xs

def sector_generator(filenames, batch_size):
    while True:
        sample = filenames[:]
        np.random.shuffle(sample)
        for i in range(0,len(sample),batch_size):
            xs = xs_from_filenames(sample[i:i+batch_size])
            ys = ys_from_filenames(sample[i:i+batch_size])
            yield xs, ys

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_file):
        self.save_file = save_file
    def on_epoch_end(self, epoch, logs):
        self.model.save(self.save_file)
        if logs['acc'] > 0.9:
            self.model.stop_training = True

if __name__ == '__main__':
    model = get_model()
    utils.load.maybe_load_weigths(model, 'model.h5')
    compile(model)
    train = utils.load.examples_from('../../datasets/carving/train')
    validation = utils.load.examples_from('../../datasets/carving/dev')
    history = model.fit_generator(sector_generator(train, 3),
        validation_data=sector_generator(validation, 10),
        validation_steps=10,
        steps_per_epoch=100,
        epochs=150,
        callbacks=[
            MyCallback('model.h5'),
            tf.keras.callbacks.TensorBoard(
                log_dir='model.tboard'
            ),
        ],
    )
