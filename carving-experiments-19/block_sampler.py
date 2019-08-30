import os
import numpy as np
import random
import math
from collections import namedtuple

class All:
    def __init__(self, folders, batch_size):
        assert type(folders) is list
        self.batch_size = batch_size
        self.filenames=[]
        for folder in folders:
            self.filenames += files_from(folder)
        self.category_func = category_from_extension
        self.sampler = sample_file_then_block
        self.categories = sorted(set([self.category_func(x) for x in self.filenames]))
        self.xs_encoder = xs_encoder_one_hot
        self.ys_encoder = mk_ys_encoder(self.categories)

    def __iter__(self):
        gen = self.sampler(self.filenames, self.category_func)
        while True:
            batch = []
            for _ in range(self.batch_size):
                sample = next(gen)
                batch.append(sample)
            xs = self.xs_encoder([s.block for s in batch])
            ys = self.ys_encoder([s.cat for s in batch])
            yield xs, ys

BlockCat = namedtuple('BlockCat', ['block', 'cat'])
def sample_file_then_block(filenames, category_func):
    while True:
        files = random.sample(filenames, len(filenames))
        for f in files:
            block = sample_sector(f)
            cat = category_func(f)
            yield BlockCat(block, cat)

def xs_encoder_one_hot(blocks):
    xs = np.zeros((len(blocks),512,256), dtype='int')
    for i,block in enumerate(blocks):
        block = np.array(block, dtype='int')
        xs[i] = one_hot(block,256)
    return xs

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
