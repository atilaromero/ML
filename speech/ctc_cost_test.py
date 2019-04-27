import tensorflow as tf
import numpy as np
from norm_gtts import norm_gtts
from ctc_loss import ctc_loss, out_chars, chars_to_ix, ix_to_chars

def test_chars_to_ix():
  assert len(out_chars)==27
  assert chars_to_ix['b'] == 1
  assert ix_to_chars[2] == 'c'


def test_ctc_loss():
  last = l0 = tf.keras.layers.Input(shape=(None,1))
  last = tf.keras.layers.LSTM(128, 
                              return_sequences=True,
                              dropout=0.5,
                              kernel_initializer=tf.keras.initializers.Ones())(last)
  last = tf.keras.layers.Dense(27, 
                               kernel_initializer=tf.keras.initializers.Ones())(last)
  last = tf.keras.layers.Activation('softmax')(last)
  model = tf.keras.Model([l0], last)
  # model.summary()

  y = 'testando'
  x = np.array([norm_gtts(y)]).reshape(1,28224,1)
  yn = np.asarray([[28224,8,*[chars_to_ix[i] for i in y]]])
  assert x.shape == (1,28224,1)
  assert yn.shape == (1, 10)

  model.compile(loss=ctc_loss(yn.shape),
    optimizer=tf.keras.optimizers.SGD(lr=0.001))

  loss = model.evaluate(x, yn)
  assert np.allclose([loss], [92922.53125])
