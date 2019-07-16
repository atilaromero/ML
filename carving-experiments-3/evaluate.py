import os
import sys
import time
import numpy as np
import itertools
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
    validation = utils.load.examples_from('../datasets/carving/dev')
    xs,ys = next(sector_generator(validation, 300, 'all')) # max is 300
    for i in range(10-1):
        xst,yst = next(sector_generator(validation, 300, 'all')) # max is 300
        xs = np.r_[xs,xst]
        ys = np.r_[ys, yst]
    zs = model.predict(xs)
    izs = np.argmax(zs, axis=1)
    iys = np.argmax(ys, axis=1)
    confyz = {}
    print(len(izs))
    for y in range(3):
        for z in range(3):
            confyz[(y,z)] = 0
    for i in range(len(zs)):
        confyz[(iys[i], izs[i])] +=1
    print('t\p\t'+'\t'.join(categories))
    for y in range(3):
        print(categories[y], end='\t')
        for z in range(3):
            print(confyz[(y,z)],end='\t')
        print()

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