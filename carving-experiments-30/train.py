import os
import sys
import time
import datetime
from collections import namedtuple
import inspect
from typing import List
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, Callback, EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.layers import Input
import models
import block_sampler
import random

# selected_models = [
#     models.D
# ]

Config = namedtuple('Config', [
    'METRIC',
    'ACTIVATION',
    'LOSS',
    'CLASSES',
    'BATCH_SIZE',
    'VALIDATION_STEPS',
    'STEPS_PER_EPOCH',
    'EPOCHS',
    'MAX_SECONDS',
    'XS_ENCODER',
    'LEN_BYTE_VECTOR',
])

def named_models():
    selected_models = []
    for n,f in inspect.getmembers(models, inspect.isfunction):
        if inspect.getfullargspec(f).args == ['classes', 'len_byte_vector', 'activation', 'loss']:
            selected_models.append(n)
    return selected_models

def main(config: Config, train_files, validation_files):
    selected_models = named_models()
    selected_models = [getattr(models, x) for x in selected_models]
    compiled_models = []
    for model in selected_models:
        m = model(config.CLASSES, len_byte_vector=config.LEN_BYTE_VECTOR, activation=config.ACTIVATION, loss=config.LOSS)
        compiled_models.append(m)
    if len(compiled_models) == 0:
        compiled_models = list(models.genall(config.CLASSES, shape=(512,config.LEN_BYTE_VECTOR)))
    print([x.name for x in compiled_models])

    time_dir=datetime.datetime.now().isoformat()[:19].replace(':','-')
    tboard_dir = os.path.join('results', time_dir, 'tboard')
    model_dir = os.path.join('results', time_dir)
    os.makedirs(model_dir)
    save_config(os.path.join(model_dir, 'config.txt'), config)
    print(config)
    results = []
    classes = block_sampler.categories_from(train_files, block_sampler.category_from_extension)
    save_experiment_results(os.path.join(model_dir, 'classes.txt'), [{"classes": ' '.join(classes)}])
    for model in compiled_models:
        model.summary()
        train_blocks = block_sampler.All(
            filenames=train_files,
            batch_size=config.BATCH_SIZE,
            xs_encoder=config.XS_ENCODER,
        )
        dev_blocks = block_sampler.All(
            filenames=validation_files,
            batch_size=config.BATCH_SIZE,
            xs_encoder=config.XS_ENCODER,
        )
        result = train(model, 
            train_generator=train_blocks,
            validation_generator=dev_blocks,
            validation_steps=config.VALIDATION_STEPS,
            steps_per_epoch=config.STEPS_PER_EPOCH,
            epochs=config.EPOCHS,
            log_dir=tboard_dir,
            model_dir=model_dir,
            max_seconds=config.MAX_SECONDS,
            metric=config.METRIC,
        )
        print(result)
        results.append(result)
        xs, ys = dev_blocks.__next__()
        confz = confusion(model, xs, ys, dev_blocks.ix_to_cat)
        print(confz)
        save_experiment_results(os.path.join(model_dir, 'confusion.txt'), confz)
    save_experiment_results(os.path.join(model_dir, 'experiments.tsv'), results)

def train(model,
        train_generator: block_sampler.All,
        validation_generator: block_sampler.All,
        validation_steps,
        steps_per_epoch,
        epochs,
        log_dir,
        model_dir,
        max_seconds,
        metric):
    start = time.time()
    history = model.fit_generator(iter(train_generator),
        validation_data=iter(validation_generator),
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[
            # MyCallback(save_file=os.path.join(model_dir, model.name + '.h5'), seconds_limit=max_seconds, metric=metric),
            MyCallback(seconds_limit=max_seconds, metric=metric),
            EarlyStopping(monitor='val_categorical_accuracy', min_delta=1e-02, patience=2),
            # TensorBoard(
            #     log_dir=os.path.join(log_dir, model.name),
            #     # update_freq=3100,
            # ),
        ],
        use_multiprocessing=False,
        workers=0,
    )
    elapsed = time.time() - start
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    epochs_count = len(history.epoch)
    val_acc = history.history['val_'+metric][-1]
    acc = history.history[metric][-1]
    m, s = divmod(elapsed, 60)
    result = {
        'Name':model.name,
        'Parameters': trainable_count,
        'Epochs': epochs_count,
        'Time': "{:d}m{:02d}s".format(int(m),int(s)),
        'Training accuracy': acc,
        'Validation accuracy': val_acc,
    }
    return result

def confusion(model, xs, ys, ix_to_cat):
    results = []
    ys_pred = model.predict(xs)
    for ix in ix_to_cat:
        result = {"True": ix_to_cat[ix]}
        ys_predix = ys_pred[np.argmax(ys, axis=-1) == ix]
        for jx in ix_to_cat:
            ypjs = ys_predix[np.argmax(ys_predix, axis=-1) == jx]
            count = len(ypjs)
            result[ix_to_cat[jx]] = count
        results.append(result)
    return results

class MyCallback(Callback):
    def __init__(self, save_file=None, seconds_limit=None, val_acc_limit=None, metric='acc', cbstep=1):
        self.seconds_limit = seconds_limit
        self.start_time = time.time()
        self.save_file = save_file
        self.val_acc_limit = val_acc_limit
        self.cbstep=cbstep
        self.metric=metric
    def on_epoch_end(self, epoch, logs):
        if epoch % self.cbstep != 0:
            return
        if self.save_file:
            self.model.save(self.save_file)
        if self.val_acc_limit and logs['val_'+self.metric] > self.val_acc_limit:
            self.model.stop_training = True
        elapsed = time.time()-self.start_time
        if self.seconds_limit and elapsed > self.seconds_limit:
            self.model.stop_training = True

def save_experiment_results(tsv_path, results):
    keys = results[0].keys()
    with open(tsv_path, 'a') as f:
        f.write('\t'.join(keys))
        f.write('\n')
        for r in results:
            values = [str(r[k]) for k in keys]
            f.write('\t'.join(values))
            f.write('\n')

def save_config(config_path, config):
    with open(config_path, 'a') as f:
        for field in config._fields:
            f.write(field)
            f.write('=')
            f.write(repr(getattr(config, field)))
            f.write('\n')

if __name__ == '__main__':
    folder = '../datasets/purified2/'
    all_files = block_sampler.files_from(folder)
    categories = {}
    for f in all_files:
        k = block_sampler.category_from_extension(f)
        categories[k] = categories.get(k, set())
        categories[k].add(f)
    sizes = [len(categories[x]) for x in categories]
    minimum = 100
    maximum = min([x for x in sizes if x >= minimum])
    print("maximum:", maximum, "minimum:", minimum)
    for k in [x for x in categories]:
        cat = categories[k]
        print(k, len(cat))
        if len(cat) < minimum:
            del categories[k]
            continue
        categories[k] = set(random.sample(cat, maximum))
    for n0 in range(20):
        for n in range(2, len(categories)+1):
            print(n0, n)
            vset = set()
            tset = set()
            sel_cat = []
            for k in random.sample(categories.keys(), n):
                cat = categories[k]
                sel_cat += [k]
                v = set(random.sample(cat, len(cat)//2))
                t = set(random.sample(cat.difference(v), len(cat)//2))
                tset = tset.union(t)
                vset = vset.union(v)
            print("train size:", len(tset), "validation size", len(vset))
            print("classes:", sel_cat)
            config = Config(
                METRIC='categorical_accuracy',
                ACTIVATION='softmax',
                LOSS='categorical_crossentropy',
                CLASSES=n,
                BATCH_SIZE=100*n,
                VALIDATION_STEPS=10,
                STEPS_PER_EPOCH=10,
                EPOCHS=10000000,
                MAX_SECONDS=10*60,
                XS_ENCODER='8bits_11',
                LEN_BYTE_VECTOR=8,
            )
            main(config, tset, vset)
