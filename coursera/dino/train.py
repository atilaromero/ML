import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

from kerasrnn import *
import tensorflow as tf
import numpy as np
import pickle
from coursera import optimize, clip
from utils import rnn_forward, rnn_backward


_data = lines2array(names, char_to_ix, ix_to_char)
X = [np.r_[np.zeros((1,27)),d][np.newaxis,:] for d in _data]
Y = [np.r_[d, [tf.keras.utils.to_categorical(0,27)]][np.newaxis,:] for d in _data]

model, getSampler = mkModel(50)

def myloss(y_true, y_pred):
    r = -tf.keras.backend.log(y_pred)
    r = y_true * r
    return tf.keras.backend.sum(r)

model.compile(loss=myloss,
              optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipvalue=5.0))

try:
    model.load_weights('model.h5')
except OSError:
    pass

for j in range(200):
    loss = 0
    for i in range(len(X[:])):
        model.reset_states()
        history = model.fit(X[i], Y[i], verbose=0)
        loss += history.history['loss'][-1]
    loss /= len(X)
    print('loss:', loss)

    model.save_weights("model.h5")

    ms = getSampler()
    for i in range(5):
        print(sample(ms))
    print()

print('done')