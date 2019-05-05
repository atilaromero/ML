import os
import sys
import numpy as np
import tensorflow as tf
from ML import ML

def get_model():
    last = l0 = tf.keras.layers.Input(shape=(512,256))
    last = tf.keras.layers.Conv1D(16, (4,), padding="same", activation="relu")(last)
    last = tf.keras.layers.MaxPooling1D(pool_size=(2,))(last)
    last = tf.keras.layers.LSTM(16, return_sequences=False, dropout=0.5,kernel_initializer=tf.keras.initializers.Ones())(last)
    last = tf.keras.layers.Dense(3)(last)
    last = tf.keras.layers.Activation('softmax')(last)

    model = tf.keras.Model([l0], last)
    return model

categories = ['pdf','png', 'jpg']
ix_to_cat = dict([(i,x) for i,x in enumerate(categories)])
cat_to_ix = dict([(x,i) for i,x in enumerate(categories)])

import utils
import utils.load
import utils.sampler

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

def train(examplesFolder, epochs=-1, sample_size=10):
    examples = utils.load.examples_from(examplesFolder)
    while(epochs != 0):
        epochs -= 1
        sample = utils.sampler.choice(examples, sample_size)
        ys = ys_from_filenames(sample)
        xs = xs_from_filenames(sample)

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

def data_generator():
    while(True):
        xs, ys = loadFolder('carving/dataset')
        yield xs, ys

def compiler(model):
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'])

if __name__ == '__main__':
    ml = ML(get_model, data_generator, save_file='carving.h5', 
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'])
    ml.main(*sys.argv)
