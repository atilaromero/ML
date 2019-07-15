import os
import sys
import time
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow.keras.backend as K

sys.path.append('..')
import utils
import utils.load
import utils.sampler

np.random.seed(1)
tf.set_random_seed(2)

def main(modelpath):
    model = tf.keras.models.load_model(modelpath)
    validation = utils.load.examples_from('../datasets/carving/dev/jpg')
    results = model.predict_generator(sector_generator(validation, 10, 'all'), steps=1)
    print(results)

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

def xs_from_filenames(filenames, blocks='all'):
    if blocks=='all':
        sampler = sample_sector
    elif blocks == 'first':
        sampler = first_sector
    else:
        raise Exception("blocks must be 'all' or 'first'")
    xs = np.zeros((len(filenames),512,256))
    for i,f in enumerate(filenames):
        x = sampler(f)
        xs[i] = utils.one_hot(x,256)
    return xs

def sector_generator(filenames, batch_size, blocks):
    while True:
        sample = filenames[:]
        np.random.shuffle(sample)
        for i in range(0,len(sample),batch_size):
            xs = xs_from_filenames(sample[i:i+batch_size], blocks)
            ys = ys_from_filenames(sample[i:i+batch_size])
            yield xs, ys


if __name__ == "__main__":
    main(*sys.argv[1:])