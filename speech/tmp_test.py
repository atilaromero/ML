import tensorflow as tf
from generateAudioSamples import generateAudioSamples

last = l0 = tf.keras.layers.Input(shape=(None,1))
# last = tf.keras.layers.Masking(mask_value=100)(last)
last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
# last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.Dense(28)(last)
last = tf.keras.layers.Activation('softmax')(last)

labels = tf.keras.layers.Input(shape=(28,))
input_length = tf.keras.layers.Input(shape=(1,))
label_length = tf.keras.layers.Input(shape=(1,))

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

last = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,))([last, labels, input_length, label_length])

model = tf.keras.Model([l0, labels, input_length, label_length], last)
model.summary()

labels = 'abcdefghijklmnopqrstuvwxyz '
assert len(labels)==27

model.compile(loss=lambda y_true, y_pred: y_pred,
    optimizer=tf.keras.optimizers.Adam(lr=0.001))

y = 'testando'
x = generateAudioSamples(y)
model.fit([x, labels, []], [y])
