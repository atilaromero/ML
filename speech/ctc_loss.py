import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

out_chars = 'abcdefghijklmnopqrstuvwxyz '
chars_to_ix = dict(zip(out_chars,range(len(out_chars))))
ix_to_chars = dict(zip(range(len(out_chars)),out_chars))

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

def to_ctc_format(xs,ys):
  max_tx = np.max([len(i) for i in xs])
  max_ty = np.max([len(i) for i in ys])
  xarr = np.zeros((len(xs), max_tx, xs[0].shape[1]))
  yarr = np.zeros((len(ys), max_ty + 2))
  for i, x in enumerate(xs):
    xarr[i,:len(x)] = x
  for i, y in enumerate(ys):
    yarr[i,:len(y)+2] = [len(x), len(y), *y]
  return xarr, yarr
