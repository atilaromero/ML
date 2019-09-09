import os
import sys
import time
import datetime
from collections import namedtuple
import inspect
from typing import List
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, Callback
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.layers import Input
import models
import block_sampler

selected_models = [
    models.D
]

Config = namedtuple('Config', [
    'METRIC',
    'ACTIVATION',
    'LOSS',
    'VALIDATION',
    'TRAIN',
    'CLASSES',
    'BATCH_SIZE',
    'VALIDATION_STEPS',
    'STEPS_PER_EPOCH',
    'EPOCHS',
    'MAX_SECONDS',
<<<<<<< HEAD
=======
    'XS_ENCODER',
    'LEN_BYTE_VECTOR',
>>>>>>> 7282463... testes progressivos
])

config = Config(
    METRIC='categorical_accuracy',
    ACTIVATION='softmax',
    LOSS='categorical_crossentropy',
<<<<<<< HEAD
    # VALIDATION = ['../datasets/govdocs1/sample/dev'],
    # TRAIN = ['../datasets/govdocs1/sample/train'],
    VALIDATION = ['../datasets/carving/dev'],
    TRAIN = ['../datasets/carving/train'],
    # CLASSES=31,
    CLASSES=3,
    BATCH_SIZE=30,
    VALIDATION_STEPS=100,
    STEPS_PER_EPOCH=100,
    EPOCHS=10000000,
    MAX_SECONDS=2*60,
=======
    VALIDATION = ['../datasets/govdocs1/sample/dev'],
    TRAIN = ['../datasets/govdocs1/sample/train'],
    # VALIDATION = ['../datasets/carving/dev'],
    # TRAIN = ['../datasets/carving/train'],
    CLASSES=31,
    # CLASSES=3,
    BATCH_SIZE=500,
    VALIDATION_STEPS=10,
    STEPS_PER_EPOCH=10,
    EPOCHS=10000000,
    MAX_SECONDS=2*60,
    XS_ENCODER='8bits_11',
    LEN_BYTE_VECTOR=8,
>>>>>>> 7282463... testes progressivos
)

def named_models():
    selected_models = []
    for n,f in inspect.getmembers(models, inspect.isfunction):
        if inspect.getfullargspec(f).args == ['classes', 'len_byte_vector', 'activation', 'loss']:
            selected_models.append(n)
    return selected_models

<<<<<<< HEAD
def gen_models(classes):
    return list(models.genall(classes, Input(shape=(512,256))))

=======
>>>>>>> 7282463... testes progressivos
def main(config: Config, *selected_models):
    selected_models = list(selected_models)
    selected_models = [getattr(models, x) for x in selected_models]
    compiled_models = []
    for model in selected_models:
<<<<<<< HEAD
        m = model(config.CLASSES, len_byte_vector=256, activation=config.ACTIVATION, loss=config.LOSS)
        compiled_models.append(m)
    if len(compiled_models) == 0:
        compiled_models = gen_models(config.CLASSES)
=======
        m = model(config.CLASSES, len_byte_vector=config.LEN_BYTE_VECTOR, activation=config.ACTIVATION, loss=config.LOSS)
        compiled_models.append(m)
    if len(compiled_models) == 0:
        compiled_models = list(models.genall(config.CLASSES, shape=(512,config.LEN_BYTE_VECTOR)))
>>>>>>> 7282463... testes progressivos
    print([x.name for x in compiled_models])

    time_dir=datetime.datetime.now().isoformat()[:19].replace(':','-')
    tboard_dir = os.path.join(time_dir, 'tboard')
    model_dir = os.path.join(time_dir)
    os.makedirs(model_dir)
    save_config(os.path.join(model_dir, 'config.txt'), config)
    print(config)
    results = []
    for model in compiled_models:
        model.summary()
        train_blocks = iter(block_sampler.All(
            folders=config.TRAIN,
            batch_size=config.BATCH_SIZE,
            xs_encoder=config.XS_ENCODER,
        ))
        dev_blocks = iter(block_sampler.All(
            folders=config.VALIDATION,
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
    save_experiment_results(os.path.join(model_dir, 'experiments.tsv'), results)

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
<<<<<<< HEAD
            MyCallback(save_file=os.path.join(model_dir, model.name + '.h5'), seconds_limit=max_seconds, metric=metric),
=======
            # MyCallback(save_file=os.path.join(model_dir, model.name + '.h5'), seconds_limit=max_seconds, metric=metric),
            MyCallback(seconds_limit=max_seconds, metric=metric),
>>>>>>> 7282463... testes progressivos
            TensorBoard(
                log_dir=os.path.join(log_dir, model.name)
            ),
        ],
<<<<<<< HEAD
=======
        use_multiprocessing=True,
>>>>>>> 7282463... testes progressivos
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
    def __init__(self, save_file=None, seconds_limit=None, val_acc_limit=None, metric='acc'):
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
    main(config, *sys.argv[1:])