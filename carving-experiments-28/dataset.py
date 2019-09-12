import os
import random

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

class Instance:
    pass

BlockCat = namedtuple('BlockCat', ['block', 'cat'])



def category_from_extension(path):
    ext = path.rsplit('.', 1)[1]
    return ext


def category_from_name(path):
    return os.path.basename(path).rsplit('.', 1)[0]


def category_from_folder(path):
    return path.rsplit('/', 2)[-2]
