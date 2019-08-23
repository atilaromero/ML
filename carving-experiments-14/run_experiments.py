import os
import sys
import time
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow.keras.backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../..')
import utils
import utils.load
import utils.sampler

def compile(model):
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['binary_accuracy', 'categorical_accuracy'])

metric="categorical_accuracy"
categories = ['txt', 'csv', 'gif', 'jpg']
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

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_file=None, seconds_limit=10*60, val_acc_limit=None):
        self.seconds_limit = seconds_limit
        self.start_time = time.time()
        self.save_file = save_file
        self.val_acc_limit = val_acc_limit
    def on_epoch_end(self, epoch, logs):
        if self.save_file:
            self.model.save(self.save_file)
        if self.val_acc_limit and logs['val_'+metric] > self.val_acc_limit:
            self.model.stop_training = True
        elapsed = time.time()-self.start_time
        if self.seconds_limit and elapsed > self.seconds_limit:
            self.model.stop_training = True

Experiment = namedtuple('Experiment', 'name model blocks')

def run_experiments(experiments,
        batch_size,
        validation_batch_size,
        validation_steps,
        steps_per_epoch,
        epochs,
        val_acc_limit=0.9):
    train = utils.load.examples_from('../dataset/train')
    assert len(train) > 0, """dataset/train contain links to govdocs1 files
    These files are not in the github repository, but they can be downloaded from
    https://digitalcorpora.org/corpora/files"""
    validation = utils.load.examples_from('../dataset/dev')
    assert len(validation) > 0, """dataset/dev contain links to govdocs1 files
    These files are not in the github repository, but they can be downloaded from
    https://digitalcorpora.org/corpora/files"""
    for e in experiments:
        compile(e.model)
        print(e.name)
        e.model.summary()
        start = time.time()
        history = e.model.fit_generator(sector_generator(train, batch_size, e.blocks),
            validation_data=sector_generator(validation, validation_batch_size, e.blocks),
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[
                MyCallback(e.name + '.h5',val_acc_limit=val_acc_limit),
                tf.keras.callbacks.TensorBoard(
                    log_dir='tboard/' + e.name
                ),
            ],
        )
        elapsed = time.time() - start
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(e.model.trainable_weights)]))
        epochs_count = len(history.epoch)
        val_acc = history.history['val_' + metric][-1]
        acc = history.history[metric][-1]
        m, s = divmod(elapsed, 60)
        yield {
            'Name':e.name,
            'Parameters': trainable_count,
            'Blocks': e.blocks,
            'Epochs': epochs_count,
            'Time': "{:d}m{:02d}s".format(int(m),int(s)),
            'Training accuracy': acc,
            'Validation accuracy': val_acc,
        }

def save_experiment_results(tsv_path, results):
    keys = results[0].keys()
    with open(tsv_path, 'a') as f:
        f.write('\t'.join(keys))
        f.write('\n')
        for r in results:
            values = [str(r[k]) for k in keys]
            f.write('\t'.join(values))
            f.write('\n')
