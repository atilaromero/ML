import tensorflow as tf
import numpy as np
from generateAudioSamples import generateAudioSamples

last = l0 = tf.keras.layers.Input(shape=(None,1))
# last = tf.keras.layers.Masking(mask_value=100)(last)
last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
# last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.Dense(27)(last)
last = tf.keras.layers.Activation('softmax')(last)

labels = tf.keras.layers.Input(shape=(8,))
input_length = tf.keras.layers.Input(shape=(1,))
label_length = tf.keras.layers.Input(shape=(1,))

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

last = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,))([last, labels, input_length, label_length])

model = tf.keras.Model([l0, labels, input_length, label_length], last)
# model.summary()

out_chars = 'abcdefghijklmnopqrstuvwxyz '
chars_to_ix = dict(zip(out_chars,range(len(out_chars))))
ix_to_chars = dict(zip(range(len(out_chars)),out_chars))
assert len(out_chars)==27
assert chars_to_ix['b'] == 1
assert ix_to_chars[2] == 'c'

model.compile(loss=lambda y_true, y_pred: y_pred,
    optimizer=tf.keras.optimizers.Adam(lr=0.001))

y = 'testando'
yn = np.asarray([chars_to_ix[i] for i in y]).reshape(1,8)
cat_yn = tf.keras.utils.to_categorical(yn,27)
x = np.array(generateAudioSamples(y)).reshape(1,28224,1)
print(x.shape)
print(yn.shape)
model.fit([x, yn, np.asarray([[28224]]), np.asarray([[8]])], yn)