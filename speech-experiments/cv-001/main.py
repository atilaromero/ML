import os
import sys
import numpy as np
import tensorflow as tf
import utils
import utils.load
import utils.sampler
import tensorflow.keras.backend as K

print("tf.VERSION", tf.VERSION)
print("tf.keras.__version__", tf.keras.__version__)

from ctc.ctc_loss import ctc_loss, to_ctc_format, ctc_predict, ix_to_chars, chars_to_ix
from audio.fft import spectrogram_from_file

def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad

def get_model():
    last = l0 = tf.keras.layers.Input(shape=(None,221))
    # last = tf.keras.layers.TimeDistributed(
    # )(last)
    last = tf.keras.layers.Conv1D(16, (3,), padding="same", activation="relu")(last)
    last = tf.keras.layers.Conv1D(8, (3,), padding="same", activation="relu")(last)
    last = tf.keras.layers.Conv1D(4, (3,), padding="same", activation="relu")(last)
    # last = tf.keras.layers.
    # last = tf.keras.layers.MaxPooling1D(pool_size=(2,))(last)
    # last = tf.keras.layers.LSTM(64, return_sequences=True)(last)
    # last = tf.keras.layers.LSTM(64, return_sequences=True)(last)
    last = tf.keras.layers.LSTM(64, return_sequences=True)(last)
    last = tf.keras.layers.Dense(27)(last)
    last = tf.keras.layers.Activation('softmax')(last)

    model = tf.keras.Model([l0], last)
    model.summary()
    return model

def compile(model, batch_size, max_ty):
    model.compile(loss=ctc_loss((batch_size, max_ty)),
        optimizer=tf.keras.optimizers.Adam())

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

def train(model, save_file, examplesFolder, batch_size=None, max_ty=100, sample_size=5, epochs=-1):
    examples = utils.load.examples_from(examplesFolder)
    sample = examples
    xs, ys = xs_ys_from_filenames(sample, max_ty)
    batch_size=batch_size and int(batch_size) or len(examples)
    accuracy = 0
    while(epochs != 0 and accuracy <0.9):
        epochs -= 1
        # sample = utils.sampler.choice(examples, sample_size)
        # xs, ys = xs_ys_from_filenames(sample, max_ty)
        w0 = model.get_weights()
        model.fit(xs,ys,batch_size=batch_size)
        if epochs%100 ==0:
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

def gradients(model, save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=5000, epochs=1):
    examples = utils.load.examples_from(examplesFolder)
    # gradient_tensors = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    # input_tensors = [model.inputs,
    #                 model.sample_weights,
    #                 model.targets,
    #                 K.learning_phase()]
    # get_gradients = K.function(inputs=input_tensors, outputs=gradient_tensors)
    while(epochs != 0):
        epochs -= 1
        sample = utils.sampler.choice(examples, sample_size)
        xs, ys = xs_ys_from_filenames(sample, max_ty)
        print(get_layer_output_grad(model, xs[:batch_size], ys[:batch_size]))
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
    def _model(save_file, examplesFolder, batch_size=None, max_ty=100):
        batch_size = batch_size and int(batch_size) or 140
        model = get_model()
        compile(model, batch_size=batch_size, max_ty=max_ty)
        utils.load.maybe_load_weigths(model, save_file=save_file)
        return model
    model = _model(*sys.argv[2:6])

    if sys.argv[1] == 'train':
        train(model, *sys.argv[2:])
    if sys.argv[1] == 'gradients':
        gradients(model, *sys.argv[2:])
    elif sys.argv[1] == 'evaluate':
        evaluate(model, *sys.argv[3:])
    elif sys.argv[1] == 'correct':
        correct(model, *sys.argv[3:])
    elif sys.argv[1] == 'failed':
        failed(model, *sys.argv[3:])
    elif sys.argv[1] == 'predictions':
        predictions(model, *sys.argv[3:])
    else:
        print(f'Use {sys.argv[0]} COMMAND')
        print(f'commands:')
        print(f'    train save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=5000, epochs=-1')
        print(f'    evaluate save_file, examplesFolder, batch_size=100, max_ty=100, sample_size=1000')
        exit(1)
