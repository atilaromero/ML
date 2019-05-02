import numpy as np
import tensorflow as tf
from load import loadFolder
from ctc_loss import to_ctc_format
from model import get_model
from ctc_loss import ctc_loss, ix_to_chars, ctc_predict
import tensorflow.keras.backend as K

sample_size = 10
model = get_model()

xs, ys = loadFolder('dataset', sample_size=sample_size)
xarr, yarr = to_ctc_format(xs, ys)
y_pred = ctc_predict(model, xarr)
for i, y in enumerate(ys):
    y_true = ''.join([ix_to_chars[j] for j in y])
    print(repr(y_true), '\t', repr(y_pred[i]))

