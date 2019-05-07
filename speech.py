import os
import sys
import numpy as np
import tensorflow as tf
import utils
import utils.load
import utils.sampler

from ctc.ctc_loss import ctc_loss, to_ctc_format, ctc_predict, ix_to_chars, chars_to_ix
from audio.fft import spectrogram_from_file

def get_model():
    last = l0 = tf.keras.layers.Input(shape=(None,221))
    # last = tf.keras.layers.Masking(mask_value=100)(last)
    # last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
    # last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.Dense(27)(last)
    last = tf.keras.layers.Activation('softmax')(last)

    model = tf.keras.Model([l0], last)
    return model

def compile(model, batch_size, max_ty):
    model.compile(loss=ctc_loss((batch_size, max_ty)),
        optimizer=tf.keras.optimizers.SGD(lr=0.001))

def xs_ys_from_filenames(filenames, max_ty):
    xs = []
    ys = []
    for f in filenames:
        y = utils.load.category_from_name(f)
        y = [chars_to_ix[j] for j in y]
        x = spectrogram_from_file(f)
        xs.append(x)
        ys.append(y)
    xs, ys = to_ctc_format(xs, ys, max_ty)
    return xs, ys

def train(model, save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=5000, epochs=-1):
    examples = utils.load.examples_from(examplesFolder)
    while(epochs != 0):
        epochs -= 1
        sample = utils.sampler.choice(examples, sample_size)
        xs, ys = xs_ys_from_filenames(sample, max_ty)
        model.fit(xs,ys,batch_size=batch_size)
        if save_file:
            model.save(save_file)

def evaluate(model, examplesFolder, batch_size=100, max_ty=100, sample_size=1000):
    examples = utils.load.examples_from(examplesFolder)
    sample = utils.sampler.choice(examples, sample_size)
    xs, ys = xs_ys_from_filenames(sample, max_ty)
    model.evaluate(xs, ys, batch_size=batch_size)

if __name__ == '__main__':
    def _train(save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=5000, epochs=-1):
        model = get_model()
        utils.load.maybe_load_weigths(model, save_file=save_file)
        compile(model, batch_size=batch_size, max_ty=max_ty)
        train(model, save_file, examplesFolder, batch_size, max_ty, sample_size, epochs)

    def _evaluate(save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=1000):
        model = get_model()
        compile(model, batch_size=batch_size, max_ty=max_ty)
        utils.load.maybe_load_weigths(model, save_file=save_file)
        evaluate(model, examplesFolder, batch_size, max_ty, sample_size)

    if sys.argv[1] == 'train':
        _train(*sys.argv[2:])
    if sys.argv[1] == 'evaluate':
        _evaluate(*sys.argv[2:])
    else:
        print(f'Use {sys.argv[0]} COMMAND')
        print(f'commands:')
        print(f'    train save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=5000, epochs=-1')
        print(f'    evaluate save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=1000')
        exit(1)
