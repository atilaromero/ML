import os
import random


class Dataset:
    def __init__(self, filenames, categories=None, category_from='extension'):
        self.filenames = set(filenames)
        if type(category_from) == str:
            assert category_from in ['extension', 'name', 'extension']
            category_from = globals()['category_from_' + category_from]
        self.category_from = category_from
        self.rebuild_categories(categories)

    def filter(self, func):
        return Dataset(filter(func, self.filenames), self.categories, self.category_from)

    def rebuild_categories(self, categories=None):
        self.categories = categories
        if self.categories == None:
            self.categories = sorted(
                set([self.category_from(x) for x in self.filenames]))
        self.cat_to_ix = dict([(x, i) for i, x in enumerate(self.categories)])
        self.ix_to_cat = dict([(i, x) for i, x in enumerate(self.categories)])

    def join(self, dataset, categories=None):
        return Dataset(self.filenames.union(dataset.filenames), categories, self.category_from)

    def clone(self, categories=None, category_from=None):
        return Dataset(self.filenames, categories or self.categories, category_from or self.category_from)

    def rnd_split_num(self, value):
        if value < 1:
            value = 1
        todo = self.filenames
        while len(todo) > 0:
            sample = random.sample(todo, min(value, len(todo)))
            todo = todo.difference(sample)
            yield Dataset(sample, self.categories, self.category_from)

    def rnd_split_fraction(self, frac):
        n = int(len(self.filenames)*frac)
        for x in self.rnd_split_num(n):
            yield x

    def by_category(self):
        assert len(self.filenames) > 0
        datasets = {}
        for f in self.filenames:
            k = self.category_from(f)
            datasets[k] = datasets.get(k, set())
            datasets[k].add(f)
        return datasets

    def filter_min_max(self, minimum=None, maximum=None):
        by_category = self.by_category()
        for k in [x for x in by_category]:
            samples = by_category[k]
            if minimum and len(samples) < minimum:
                del by_category[k]
                continue
            if maximum and len(samples) > maximum:
                by_category[k] = set(random.sample(samples, maximum))
        filenames = set()
        for v in by_category.values():
            filenames = filenames.union(v)
        return Dataset(filenames, by_category.keys(), category_from=self.category_from)

    @classmethod
    def new_from_folders(cls, *folders, categories=None, category_from='extension'):
        result = set()
        for folder in folders:
            for dirpath, _, filenames in os.walk(folder):
                for f in filenames:
                    result.add(os.path.join(dirpath, f))
        return Dataset(result, categories, category_from)


def category_from_extension(path):
    ext = path.rsplit('.', 1)[1]
    return ext


def category_from_name(path):
    return os.path.basename(path).rsplit('.', 1)[0]


def category_from_folder(path):
    return path.rsplit('/', 2)[-2]
