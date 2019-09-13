import os
import math
import random
import numpy as np
from typing import Generator
from dataset import Dataset
from abc import ABC, abstractmethod

class BlockInstance:
    def __init__(self, block, category):
        self.block = block
        self.category = category


class BlockSampler:
    def __init__(self, dataset: Dataset, group_by='by_file'):
        self.dataset = dataset
        assert group_by in ['by_file', 'by_sector']
        self.group_by = group_by
    def __iter__(self) -> Generator[BlockInstance, None, None]:
        filenames = list(self.dataset.filenames)
        assert len(filenames) > 0
        sectors = {}
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        while True:
            if self.group_by == 'by_file':
                files = random.sample(filenames, len(filenames))
            if self.group_by == 'by_sector':
                files = random.choices(*zip(*sectors.items()), k=1000)
            for f in files:
                sector = random.randrange(sectors[f])
                block = get_sector(f, sector)
                yield BlockInstance(block, self.dataset.categories_from(f))

class BlockSamplerByFile:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    def __iter__(self) -> Generator[BlockInstance, None, None]:
        filenames = list(self.dataset.filenames)
        assert len(filenames) > 0
        sectors = {}
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        while True:
            files = random.sample(filenames, len(filenames))
            for f in files:
                sector = random.randrange(sectors[f])
                block = get_sector(f, sector)
                yield BlockInstance(block, self.dataset.categories_from(f))


class BlockSamplerBySector:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    def __iter__(self) -> Generator[BlockInstance, None, None]:
        filenames = list(self.dataset.filenames)
        assert len(filenames) > 0
        sectors = {}
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        while True:
            files = random.choices(*zip(*sectors.items()), k=1000)
            for f in files:
                sector = random.randrange(sectors[f])
                block = get_sector(f, sector)
                yield BlockInstance(block, self.dataset.categories_from(f))

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
