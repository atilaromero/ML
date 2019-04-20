import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# model0 = tf.keras.models.load_model('models/model_shakespeare_kiank_350_epoch.h5')

# model0.summary()
# with open('model0.json', 'w') as f:
#     f.write(model0.to_json())

last = l0 = tf.keras.layers.Input(
    shape=(40, 38), 
    name='input_3')
last = tf.keras.layers.LSTM(
    128, 
    name='lstm_5',
    return_sequences=True, kernel_initializer=tf.keras.initializers.VarianceScaling(
        mode='fan_avg',
        distribution='uniform'))(last)
last = tf.keras.layers.Dropout(
    0.5, 
    name='dropout_3')(last)
last = tf.keras.layers.LSTM(
    128,
    name='lstm_6',
    return_sequences=False, kernel_initializer=tf.keras.initializers.VarianceScaling(
        mode='fan_avg',
        distribution='uniform'))(last)
last = tf.keras.layers.Dropout(
    0.5,
    name='dropout_4')(last)
last = tf.keras.layers.Dense(
    38,
    name='dense_3',
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        mode='fan_avg',
        distribution='uniform'))(last)
last = tf.keras.layers.Activation(
    'softmax',
    name='activation_3')(last)

model2 = tf.keras.Model([l0], last, name='model_1')

# model2.summary()

# with open('model2.json', 'w') as f:
#     f.write(model2.to_json())

import io
import sys
import numpy as np

def load_data(path, maxlen):
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
    # print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    # print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    # print('nb sequences:', len(sentences))

    # print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y, char_indices, indices_char, text, chars

maxlen = 40
x, y, char_indices, indices_char, text, chars = load_data('shakespeare.txt', maxlen)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = np.random.randint(0, len(text) - maxlen - 1)
    for diversity in [1.0]:# [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(100):# (400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model2.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
save_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda *args: model2.save_weights("model.h5"))

model2.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.RMSprop(lr=0.01))

try:
    model2.load_weights('model.h5')
except OSError:
    pass

model2.fit(x, y,
        batch_size=128,
        epochs=60,
        callbacks=[save_callback, print_callback])
    

