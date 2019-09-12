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
    def __init__(self, filenames, categories=None, categories_from='extension', distribution='by_file'):
        self.filenames = set(filenames)
        if type(categories_from) == str:
            assert categories_from in ['extension', 'name', 'extension']
            categories_from = globals()['categories_from_' + categories_from]
        self.categories_from = categories_from
        assert distribution in ['by_file', 'by_sector']
        self.distribution = distribution
        self.rebuild_categories(categories)

    def filter(self, func):
        return Dataset(filter(func, self.filenames), self.categories, self.categories_from, self.distribution)

    def rebuild_categories(self, categories=None):
        self.categories = categories
        if self.categories == None:
            self.categories = sorted(
                set([self.categories_from(x) for x in self.filenames]))
        self.cat_to_ix = dict([(x, i) for i, x in enumerate(self.categories)])
        self.ix_to_cat = dict([(i, x) for i, x in enumerate(self.categories)])

    def join(self, dataset, categories=None):
        return Dataset(self.filenames.union(dataset.filenames), categories, self.categories_from, self.distribution)
    
    def clone(self, categories=None, categories_from=None, distribution=None):
        return Dataset(self.filenames, categories or self.categories, categories_from or self.categories_from, distribution or self.distribution)

    def rnd_split_num(self, value):
        if value < 1:
            value = 1
        todo = self.filenames
        while len(todo) > 0:
            sample = random.sample(todo, min(value, len(todo)))
            todo = todo.difference(sample)
            yield Dataset(sample, self.categories, self.categories_from, self.distribution)

    def rnd_split_fraction(self, frac):
        n = int(len(self.filenames)*frac)
        for x in self.rnd_split_num(n):
            yield x

    def by_category(self):
        datasets = {}
        for f in self.filenames:
            k = self.categories_from(f)
            datasets[k] = datasets.get(k, set())
            datasets[k].add(f)
        return datasets

    def generator(self) -> Generator[BlockInstance, None, None]:
        """ generator of BlockInstance"""
        filenames = list(self.filenames)
        assert len(filenames) > 0
        sectors = {}
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        while True:
            if self.distribution == 'by_file':
                files = random.sample(filenames, len(filenames))
            if self.distribution == 'by_sector':
                files = random.choices(*zip(*sectors.items()), k=1000)
            for f in files:
                sector = random.randrange(sectors[f])
                block = get_sector(f, sector)
                yield BlockInstance(block, self.categories_from(f))

    @classmethod
    def new_from_folders(cls, *folders, categories=None, categories_from='extension', distribution='by_file'):
        result = set()
        for folder in folders:
            for dirpath, _, filenames in os.walk(folder):
                for f in filenames:
                    result.add(os.path.join(dirpath, f))
        return Dataset(result, categories, categories_from, distribution)


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


def categories_from_extension(path):
    ext = path.rsplit('.', 1)[1]
    return ext


def categories_from_name(path):
    return os.path.basename(path).rsplit('.', 1)[0]


def categories_from_folder(path):
    return path.rsplit('/', 2)[-2]
