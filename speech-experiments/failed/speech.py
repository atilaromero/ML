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
    last = tf.keras.layers.LSTM(128, return_sequences=True)(last)
    last = tf.keras.layers.Dense(27)(last)
    last = tf.keras.layers.Activation('softmax')(last)

    model = tf.keras.Model([l0], last)
    return model

def compile(model, batch_size, max_ty):
    model.compile(loss=ctc_loss((batch_size, max_ty)),
        optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.5, clipnorm=1.0))

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

def train(model, save_file, examplesFolder, batch_size=140, max_ty=100, sample_size=140, epochs=-1):
    examples = utils.load.examples_from(examplesFolder)
    sample = examples
    xs, ys = xs_ys_from_filenames(sample, max_ty)
    while(epochs != 0):
        epochs -= 1
        # sample = utils.sampler.choice(examples, sample_size)
        # xs, ys = xs_ys_from_filenames(sample, max_ty)
        w0 = model.get_weights()
        model.fit(xs,ys,batch_size=batch_size,shuffle=True)
        if epochs%100 == 0:
            w1 = model.get_weights()
            for l0,l1 in zip(w0,w1):
                diff = l1-l0
                sadiff = np.sum(np.abs(diff))
                saw1 = np.sum(np.abs(l1))
                print('sum(abs(grads))','%1.5E'%sadiff,
                    'sum(grads)','%1.5E'%np.sum(diff),
                    'sum(abs(weights))', '%1.5E'%saw1, 
                    'zeros', len(np.where(l1 == 0)), 
                    l1.shape)
            if save_file:
                model.save(save_file)
            accuracy = get_accuracy(model, sample[:100], xs[:100])
            print('accuracy: %0.2f' % accuracy)

def predict(model, sample, xs):
    y_pred = ctc_predict(model, xs)
    y_true = [utils.load.category_from_name(f) for f in sample]
    return y_true, y_pred

def get_accuracy(model, sample, xs):
    y_true, y_pred = predict(model, sample, xs)
    matches = [i==j for i,j in zip(y_pred, y_true)]
    accuracy = sum(matches)/len(matches)
    return accuracy

def evaluate(model, examplesFolder, batch_size=100, max_ty=100, sample_size=1000):
    examples = utils.load.examples_from(examplesFolder)
    sample = utils.sampler.choice(examples, sample_size)
    xs, ys = xs_ys_from_filenames(sample, max_ty)
    accuracy = get_accuracy(model, sample, xs)
    print('accuracy: %0.2f' % accuracy)

def correct(model, examplesFolder, batch_size=100, max_ty=100):
    examples = utils.load.examples_from(examplesFolder)
    xs, ys = xs_ys_from_filenames(examples, max_ty)
    y_true, y_pred = predict(model, examples, xs)
    for t,p in zip(y_true, y_pred):
        if t == p:
            print(repr(t), repr(p), )

def predictions(model, examplesFolder, batch_size=100, max_ty=100):
    examples = utils.load.examples_from(examplesFolder)
    xs, ys = xs_ys_from_filenames(examples, max_ty)
    y_pred = model.predict(xs,batch_size=batch_size)
    print(y_pred)

def failed(model, examplesFolder, batch_size=100, max_ty=100):
    examples = utils.load.examples_from(examplesFolder)
    xs, ys = xs_ys_from_filenames(examples, max_ty)
    y_true, y_pred = predict(model, examples, xs)
    for t,p in zip(y_true, y_pred):
        if t != p:
            print(repr(t), repr(p))

if __name__ == '__main__':
    def _model(save_file, examplesFolder, batch_size=140, max_ty=100):
        model = get_model()
        compile(model, batch_size=batch_size, max_ty=max_ty)
        utils.load.maybe_load_weigths(model, save_file=save_file)
        return model
    model = _model(*sys.argv[2:6])

    if sys.argv[1] == 'train':
        train(model, *sys.argv[2:])
    elif sys.argv[1] == 'evaluate':
        evaluate(model, *sys.argv[3:])
    elif sys.argv[1] == 'correct':
        correct(model, *sys.argv[3:])
    elif sys.argv[1] == 'failed':
        failed(model, *sys.argv[3:])
    elif sys.argv[1] == 'predictions':
        predictions(model, *sys.argv[3:])
    else:
        print('Use %s{sys.argv[0]} COMMAND')
        print('commands:')
        print('    train save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=5000, epochs=-1')
        print('    evaluate save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=1000')
        exit(1)
