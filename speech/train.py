import numpy as np
import tensorflow as tf
from load import loadFolder
from ctc_loss import to_ctc_format
from model import get_model
from ctc_loss import ctc_loss, ctc_predict, ix_to_chars


sample_size = 5000
batch_size = 100
max_ty = 100
model = get_model()
model.compile(loss=ctc_loss((batch_size, max_ty)),
    optimizer=tf.keras.optimizers.SGD(lr=0.001))

while(True):
    xs, ys = loadFolder('dataset', sample_size=sample_size)
    xarr, yarr = to_ctc_format(xs, ys, max_ty=100)
    model.fit(xarr,yarr,batch_size=batch_size)
    model.save('model.h5')
    y_pred = ctc_predict(model, xarr[:5])
    for i, y in enumerate(ys[:5]):
        y_true = ''.join([ix_to_chars[j] for j in y])
        print(repr(y_true), '\t', repr(y_pred[i]))