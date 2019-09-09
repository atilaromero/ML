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
    confusions = []
    for model in compiled_models:
        model.summary()
        train_blocks = iter(block_sampler.All(
            filenames=train_files,
            batch_size=config.BATCH_SIZE,
            xs_encoder=config.XS_ENCODER,
        ))
        dev_blocks = iter(block_sampler.All(
            filenames=validation_files,
            batch_size=config.BATCH_SIZE,
            xs_encoder=config.XS_ENCODER,
        ))
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
        confusions.append(confusion(model, dev_blocks))
    save_experiment_results(os.path.join(model_dir, 'experiments.tsv'), results)
    save_experiment_results(os.path.join(model_dir, 'confusion.txt'), confusions)

def confusion(model, generator):
    xs, ys = next(iter(generator))
    xs0 = xs[ys[:,0]==1]
    ys0 = ys[ys[:,0]==1]
    xs1 = xs[ys[:,1]==1]
    ys1 = ys[ys[:,1]==1]
    yp0 = model.predict(xs0)
    yp1 = model.predict(xs1)
    yp0t = sum(np.argmax(yp0, axis=-1)==np.argmax(ys0, axis=-1))
    yp0f = sum(np.argmax(yp0, axis=-1)!=np.argmax(ys0, axis=-1))
    yp1t = sum(np.argmax(yp1, axis=-1)==np.argmax(ys1, axis=-1))
    yp1f = sum(np.argmax(yp1, axis=-1)!=np.argmax(ys1, axis=-1))
    result = {
        'file': yp0t/len(yp0),
        'random': yp1t/len(yp1),
        'predicted_randomness': (1-(yp0t/len(yp0)))/(yp1t/len(yp1))
    }
    print(result)
    return result

def train(model,
        train_generator,
        validation_generator,
        validation_steps,
        steps_per_epoch,
        epochs,
        log_dir,
        model_dir,
        max_seconds,
        metric):

    start = time.time()
    history = model.fit_generator(train_generator,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[
            # MyCallback(save_file=os.path.join(model_dir, model.name + '.h5'), seconds_limit=max_seconds, metric=metric),
            MyCallback(seconds_limit=max_seconds, metric=metric),
            EarlyStopping(patience=3),
            TensorBoard(
                log_dir=os.path.join(log_dir, model.name),
                # update_freq=3100,
            ),
        ],
        use_multiprocessing=True,
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
    folder = '/home/user/sample3'
    all_files = block_sampler.files_from(folder)
    rnd_files = block_sampler.files_from('/home/user/random')
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

    for k in ['java', 'pdf']: #categories:
        cat = categories[k]
        mix = cat.union(rnd_files)
        v = set(random.sample(mix, len(mix)//2))
        t = set(random.sample(mix.difference(v), len(mix)//2))
        tset = t
        vset = v
        print("train size:", len(tset), "validation size", len(vset))
        print("classes:", [k, 'rnd'])
        config = Config(
            METRIC='categorical_accuracy',
            ACTIVATION='softmax',
            LOSS='categorical_crossentropy',
            CLASSES=2,
            BATCH_SIZE=500,
            VALIDATION_STEPS=10,
            STEPS_PER_EPOCH=10,
            EPOCHS=10000000,
            MAX_SECONDS=2*60,
            XS_ENCODER='8bits_11',
            LEN_BYTE_VECTOR=8,
        )
        main(config, tset, vset)