import io
import tensorflow as tf
from generate_output import generate_output
from build_data import build_data
from vectorization import vectorization
from sample import sample
import numpy as np

print("Loading text data...")
text = io.open('shakespeare.txt', encoding='utf-8').read().lower()
#print('corpus length:', len(text))

Tx = 40
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print('number of unique characters in the corpus:', len(chars))

print("Creating training set...")
X, Y = build_data(text, Tx, stride = 3)
print("Vectorizing training set...")
x, y = vectorization(X, Y, n_x = len(chars), char_indices = char_indices) 
print("Loading model...")
model = tf.keras.models.load_model('models/model_shakespeare_kiank_350_epoch.h5')

def on_epoch_end(epoch, logs):
    pass

print_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

generate_output(Tx, chars, char_indices, indices_char, model, sample)