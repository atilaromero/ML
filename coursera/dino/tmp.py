from kerasrnn import *
import tensorflow as tf
import numpy as np
import pickle
from coursera import optimize
from utils import rnn_forward


def test_loss():
    np.random.seed(1)
    vocab_size, n_a = 27, 100
    a_prev = np.zeros((n_a, 1))
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    X = [12,3,5,11,22,3]
    Y = [4,14,11,22,25, 26]
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

    X2 = [tf.keras.utils.to_categorical(i, 27) for i in X]
    X2 = np.array(X2)[np.newaxis,:]
    Y2 = [tf.keras.utils.to_categorical(i, 27) for i in Y]
    Y2 = np.array(Y2)[np.newaxis,:]

    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    assert np.allclose(loss, 74.08235894494784)

    model, getStep = mkModel(100)
    model.set_weights((Wax.T, Waa.T, b.T[0], Wya.T, by.T[0]))
    step = getStep()
    Y_2 = model.predict(X2)
    loss2 = -np.log(Y_2)
    loss2[Y2 == 0] = 0
    loss2 = np.sum(loss2)
    assert np.allclose(loss2, loss)

    def myloss(y_true, y_pred):
        r = -tf.keras.backend.log(y_pred)
        r = y_true * r
        return tf.keras.backend.sum(r)

    model.compile(loss=myloss,
              optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipvalue=5.0))
    loss3 = model.evaluate(X2, Y2, verbose=0)
    # model = tf.keras.Model()
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    print(grads)
    assert 1 ==2
    assert np.allclose(loss3, loss)
