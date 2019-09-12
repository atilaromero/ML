import os
import math
import random
import numpy as np
from typing import Generator

class BlockInstance:
    def __init__(self, block, category):
        self.block = block
        self.category = category

class Dataset:
    def __init__(self, filenames, categories=None):
        self.filenames = set(filenames)
        self.category_func = category_from_extension
        self.rebuild_categories(categories)

    def filter(self, func):
        return Dataset(filter(func, self.filenames), self.categories)

    def rebuild_categories(self, categories=None):
        self.categories = categories
        if self.categories == None:
            self.categories = sorted(
                set([self.category_func(x) for x in self.filenames]))
        self.cat_to_ix = dict([(x, i) for i, x in enumerate(self.categories)])
        self.ix_to_cat = dict([(i, x) for i, x in enumerate(self.categories)])

    def join(self, dataset, categories=None):
        return Dataset(self.filenames.union(dataset.filenames), categories=categories)

    def rnd_split_num(self, value):
        if value < 1:
            value = 1
        todo = self.filenames
        while len(todo) > 0:
            sample = random.sample(todo, min(value, len(todo)))
            todo = todo.difference(sample)
            yield Dataset(sample, self.categories)

    def rnd_split_fraction(self, frac):
        n = int(len(self.filenames)*frac)
        for x in self.rnd_split_num(n):
            yield x

    def by_category(self):
        datasets = {}
        for f in self.filenames:
            k = self.category_func(f)
            datasets[k] = datasets.get(k, set())
            datasets[k].add(f)
        return datasets

    def generator(self, distribution='by_file') -> Generator[BlockInstance, None, None]:
        """ generator of BlockInstance
        distribution = 'by_file' or 'by_sector'"""
        assert distribution in ['by_file', 'by_sector']
        filenames = list(self.filenames)
        assert len(filenames) > 0
        sectors = {}
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        while True:
            if distribution=='by_file':
                files = random.sample(filenames, len(filenames))
            if distribution=='by_sector':
                files = random.choices(*zip(*sectors.items()), k=1000)
            for f in files:
                sector = random.randrange(sectors[f])
                block = get_sector(f, sector)
                yield BlockInstance(block, self.category_func(f))


    @classmethod
    def new_from_folders(cls, folders):
        result = set()
        for folder in folders:
            for dirpath, _, filenames in os.walk(folder):
                for f in filenames:
                    result.add(os.path.join(dirpath, f))
        return Dataset(result)
    @classmethod
    def new_from_folder(cls, folder):
        return Dataset.new_from_folders([folder])

def count_sectors(filename):
    stat = os.stat(filename)
    return math.ceil(stat.st_size/512)

def get_sector(filename, sector):
    with open(filename, 'rb') as f:
        f.seek(sector*512, 0)
        b = f.read(512)
        assert len(b) > 0
        n = np.zeros((512), dtype='int')
        n[:len(b)] = [int(x) for x in b]
        return n

def category_from_extension(path):
    ext = path.rsplit('.', 1)[1]
    return ext


def category_from_name(path):
    return os.path.basename(path).rsplit('.', 1)[0]


def category_from_folder(path):
    return path.rsplit('/', 2)[-2]
