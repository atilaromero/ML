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
    model.compile(loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])

categories = ['pdf','png', 'jpg']
ix_to_cat = dict([(i,x) for i,x in enumerate(categories)])
cat_to_ix = dict([(x,i) for i,x in enumerate(categories)])

def two_sectors(path):
    with open(path, 'rb') as f:
        f.seek(0,2)
        size = f.tell()
        i = np.random.randint(0,size)
        sector = i//512
        f.seek(sector*512,0)
        b = f.read(512)
        n1 = np.zeros((512),dtype='int')
        n1[:len(b)] = [int(x) for x in b]
        b = f.read(512)
        n2 = np.zeros((512),dtype='int')
        n2[:len(b)] = [int(x) for x in b]
        return n1,n2

def xs_ys_from_filenames(filenames):
    xs = np.zeros((len(filenames),1024,256))
    ys = np.random.choice([0,1],len(filenames))
    for i,f in enumerate(filenames):
        if ys[i]: # blocks should be consecutive
            n1, n2 = two_sectors(f)
        else: # blocks should not be consecutive
            n1, _ = two_sectors(f)
            n2, _ = two_sectors(f)
        xs[i,0:512] = utils.one_hot(n1,256)
        xs[i,512:1024] = utils.one_hot(n2,256)
    return xs, ys

def sector_generator(filenames, batch_size):
    while True:
        sample = filenames[:]
        assert len(sample) > 0
        np.random.shuffle(sample)
        for i in range(0,len(sample),batch_size):
            xs, ys = xs_ys_from_filenames(sample[i:i+batch_size])
            yield xs, ys

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_file=None, seconds_limit=10*60):
        self.seconds_limit = seconds_limit
        self.start_time = time.time()
        self.save_file = save_file
    def on_epoch_end(self, epoch, logs):
        if self.save_file:
            self.model.save(self.save_file)
        if logs['acc'] > 0.9:
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
    train = utils.load.examples_from('../datasets/carving/train')
    validation = utils.load.examples_from('../datasets/carving/dev')
    for e in experiments:
        if os.path.exists(e.name+'.h5'):
            os.remove(e.name+'.h5')
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
                MyCallback(e.name + '.h5'),
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
