import tensorflow as tf
import numpy as np
from generateAudioSamples import generateAudioSamples
import tensorflow.keras.backend as K

last = l0 = tf.keras.layers.Input(shape=(None,1))
# last = tf.keras.layers.Masking(mask_value=100)(last)
# last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
# last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.Dense(27)(last)
last = tf.keras.layers.Activation('softmax')(last)

def ctc_loss(y_shape):
  def f(y_true, y_pred):
    y_true = tf.reshape(y_true, y_shape)
    k_inputs = y_pred
    k_input_lens = y_true[:,0:1]
    k_label_lens = y_true[:,1:2]
    k_labels = y_true[:,2:]
    cost = K.ctc_batch_cost(k_labels, k_inputs, k_input_lens,k_label_lens)
    return cost
  return f


model = tf.keras.Model([l0], last)
# model.summary()

out_chars = 'abcdefghijklmnopqrstuvwxyz '
chars_to_ix = dict(zip(out_chars,range(len(out_chars))))
ix_to_chars = dict(zip(range(len(out_chars)),out_chars))

def test_A():
  assert len(out_chars)==27
  assert chars_to_ix['b'] == 1
  assert ix_to_chars[2] == 'c'


def test_B():
  y = 'testando'
  x = np.array([generateAudioSamples(y)]).reshape(1,28224,1)
  yn = np.asarray([[28224,8,*[chars_to_ix[i] for i in y]]])
  assert x.shape == (1,28224,1)
  assert yn.shape == (1, 10)

  model.compile(loss=ctc_loss(yn.shape),
    optimizer=tf.keras.optimizers.SGD(lr=0.001))

  loss = model.evaluate(x, yn)
  assert np.allclose([loss], [0])
