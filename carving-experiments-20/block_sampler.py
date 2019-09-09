import os
import numpy as np
import random
import math
import threading
from collections import namedtuple

def xs_encoder_one_hot(blocks):
    xs = np.zeros((len(blocks),512,256), dtype='int')
    for i,block in enumerate(blocks):
        block = np.array(block, dtype='int')
        xs[i] = one_hot(block,256)
    return xs

def xs_encoder_264bits(blocks):
    xs = np.zeros((len(blocks),512,264), dtype='int')
    xs[:,:,:256] = xs_encoder_one_hot(blocks)
    xs[:,:,256:] = xs_encoder_8bits_11(blocks)
    return xs


bitmap = np.array([128,64,32,16,8,4,2,1], dtype='int').reshape((1,8)).repeat(512,0)
def xs_encoder_8bits01(blocks):
    xs = np.zeros((len(blocks),512,8), dtype='int')
    for i,block in enumerate(blocks):
        blk = block.reshape((512,1)).repeat(8,1)
        bits = np.bitwise_and(blk, bitmap)/bitmap
        xs[i] = bits
    return xs

def xs_encoder_8bits_11(blocks):
    xs = xs_encoder_8bits01(blocks)
    xs = xs * 2 -1
    return xs

def xs_encoder_16bits(blocks):
    xs = np.zeros((len(blocks),512,16), dtype='int')
    xs8 = xs_encoder_8bits01(blocks)
    xs[:,:,:8] = xs8
    xs[:,:,8:] = 1 - xs8
    return xs

class All:
    def __init__(self, folders, batch_size, xs_encoder=xs_encoder_one_hot):
        assert type(folders) is list
        self.lock = threading.Lock()
        self.batch_size = batch_size
        self.filenames=[]
        for folder in folders:
            self.filenames += files_from(folder)
        self.category_func = category_from_extension
        self.sampler = sample_file_then_block
        self.categories = sorted(set([self.category_func(x) for x in self.filenames]))
        if type(xs_encoder) == str:
            xs_encoder = globals()['xs_encoder_' + xs_encoder]
        self.xs_encoder = xs_encoder
        self.ys_encoder = mk_ys_encoder(self.categories)
        self.gen = self.sampler(self.filenames, self.category_func)

    def __iter__(self):
        while True:
            xs, ys = next(self)
            yield xs, ys

    def __next__(self):
        with self.lock:
            batch = []
            for _ in range(self.batch_size):
                sample = next(self.gen)
                batch.append(sample)
            xs = self.xs_encoder([s.block for s in batch])
            ys = self.ys_encoder([s.cat for s in batch])
            return xs, ys


    def __myiter__(self):
        gen = self.sampler(self.filenames, self.category_func)
        while True:
            batch = []
            for _ in range(self.batch_size):
                sample = next(gen)
                batch.append(sample)
            xs = self.xs_encoder([s.block for s in batch])
            ys = self.ys_encoder([s.cat for s in batch])
            yield xs, ys

class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

BlockCat = namedtuple('BlockCat', ['block', 'cat'])
def sample_file_then_block(filenames, category_func):
    while True:
        files = random.sample(filenames, len(filenames))
        assert len(files) > 0
        for f in files:
            block = sample_sector(f)
            cat = category_func(f)
            yield BlockCat(block, cat)

def mk_ys_encoder(allcategories):
    cat_to_ix = dict([(x,i) for i,x in enumerate(allcategories)])
    def ys_encoder(cats):
        ys = np.zeros((len(cats), len(allcategories)), dtype='int')
        for i,cat in enumerate(cats):
            y = cat_to_ix[cat]
            ys[i] = one_hot(y,len(allcategories))
        return ys
    return ys_encoder

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

def files_from(folder):
    def inner():
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                yield os.path.join(dirpath, f)
    return list(inner())

def category_from_extension(path):
    ext = path.rsplit('.',1)[1]
    return ext

def category_from_name(path):
    return os.path.basename(path).rsplit('.',1)[0]

def category_from_folder(path):
    return path.rsplit('/',2)[-2]

def one_hot(arr, num_categories):
    arr_shape = np.shape(arr)
    flatten = np.reshape(arr, -1)
    r = np.zeros((len(flatten),num_categories))
    r[np.arange(len(flatten)),flatten] = 1
    return r.reshape((*arr_shape,num_categories))
# def one_hot(arr, num_categories):
    # return np.eye(num_categories)[arr]
