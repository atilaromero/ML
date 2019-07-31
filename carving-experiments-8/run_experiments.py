import os
import sys
import time
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow.keras.backend as K
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('..')
import utils
import utils.load
import utils.sampler

def compile(model):
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])

categories = ['pdf','png', 'jpg']
ix_to_cat = dict([(i,x) for i,x in enumerate(categories)])
cat_to_ix = dict([(x,i) for i,x in enumerate(categories)])

def sample_sector(path, fill_random):
    count = count_sectors(path)
    sector = np.random.randint(0,count)
    x, y = get_sector(path, sector, fill_random)
    return x, y

def first_sector(path, fill_random):
    x, y = get_sector(path, 0, fill_random)
    return x, y

def last_sector(path, fill_random):
    count = count_sectors(path)
    x, y = get_sector(path, count-1, fill_random)
    return x, y

def get_sector(path, sector, fill_random):
    x = np.zeros((512), dtype='int')
    y = np.zeros((512))
    with open(path, 'rb') as f:
        f.seek(sector*512,0)
        b = f.read(512)
        x[:len(b)] = [int(x) for x in b]
        if fill_random:
            x[len(b):] = np.random.randint(256, size=(512-len(b)))
        y[:len(b)] = 1
        return x, y

def count_sectors(path):
    with open(path, 'rb') as f:
        f.seek(0,2)
        size = f.tell()
        return math.ceil(size/512.0)

def xs_ys_from_filenames(filenames):
    ys = np.zeros((len(filenames),512))
    xs = np.zeros((len(filenames),512,256))
    for i,f in enumerate(filenames):
        sampler = np.random.choice([sample_sector, last_sector], p=[0.5,0.5])
        x, y = sampler(f, fill_random=True)
        xs[i] = utils.one_hot(x,256)
        ys[i] = y
    return xs, ys

def sector_generator(filenames, batch_size):
    while True:
        sample = filenames[:]
        np.random.shuffle(sample)
        for i in range(0,len(sample),batch_size):
            xs, ys = xs_ys_from_filenames(sample[i:i+batch_size])
            yield xs, ys

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_file=None, seconds_limit=10*60, acc_limit=0.9):
        self.seconds_limit = seconds_limit
        self.start_time = time.time()
        self.save_file = save_file
        self.acc_limit = acc_limit
    def on_epoch_end(self, epoch, logs):
        if self.save_file:
            self.model.save(self.save_file)
        if self.acc_limit and logs['acc'] > self.acc_limit:
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
        epochs):
    train = utils.load.examples_from('../datasets/carving/train/pdf')
    validation = utils.load.examples_from('../datasets/carving/dev/pdf')
    for e in experiments:
        compile(e.model)
        print(e.name)
        e.model.summary()
        start = time.time()
        history = e.model.fit_generator(sector_generator(train, batch_size),
            validation_data=sector_generator(validation, validation_batch_size),
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[
                MyCallback(e.name + '.h5',seconds_limit=10*60, acc_limit=None),
                tf.keras.callbacks.TensorBoard(
                    log_dir='tboard/' + e.name
                ),
            ],
        )
        elapsed = time.time() - start
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(e.model.trainable_weights)]))
        epochs_count = len(history.epoch)
        val_acc = history.history['val_acc'][-1]
        acc = history.history['acc'][-1]
        m, s = divmod(elapsed, 60)
        yield {
            'Name':e.name,
            'Parameters': trainable_count,
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
