import tensorflow as tf
import numpy as np
from generateAudioSamples import generateAudioSamples
import tensorflow.keras.backend as K

last = l0 = tf.keras.layers.Input(shape=(None,1))
# last = tf.keras.layers.Masking(mask_value=100)(last)
last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
# last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.Dense(27)(last)
last = tf.keras.layers.Activation('softmax')(last)

def ctc_loss(y_true, y_pred):
    k_labels = y_true
    k_inputs = y_pred
    k_input_lens = tf.expand_dims(tf.expand_dims(tf.shape(y_pred)[1],0),0)
    k_label_lens = tf.expand_dims(tf.expand_dims(tf.shape(y_true)[1],0),0)
    return K.ctc_batch_cost(k_labels, k_inputs, k_input_lens,
                              k_label_lens)[0]

def test_A():
    y_true = K.variable(np.asarray([[0,1]]), dtype='int32')
    y_pred = K.variable(np.asarray(tf.keras.utils.to_categorical([[0,1]])), dtype='float32')
    cost = K.eval(ctc_loss(y_true,y_pred))
    print(cost)
    assert cost[0]**2**0.5 < 1e-5
test_A()

model = tf.keras.Model([l0], last)
# model.summary()

out_chars = 'abcdefghijklmnopqrstuvwxyz '
chars_to_ix = dict(zip(out_chars,range(len(out_chars))))
ix_to_chars = dict(zip(range(len(out_chars)),out_chars))
assert len(out_chars)==27
assert chars_to_ix['b'] == 1
assert ix_to_chars[2] == 'c'

model.compile(loss=ctc_loss,
    optimizer=tf.keras.optimizers.Adam(lr=0.001))

y = 'testando'
yn = np.asarray([chars_to_ix[i] for i in y])
cat_yn = tf.keras.utils.to_categorical(yn,27)
print(yn)
x = np.array(generateAudioSamples(y))
print(x.shape)
# model.fit(x.reshape(1,28224,1), yn.reshape(1,8,27))