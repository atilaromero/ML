from kerasrnn import *
import pickle
from coursera import optimize, clip
from utils import rnn_forward, rnn_backward

def test_version():
    assert tf.VERSION == '1.13.1'
    assert tf.keras.__version__ == '2.2.4-tf'

_data = lines2array(names, char_to_ix, ix_to_char)
X = [np.r_[np.zeros((1,27)),d][np.newaxis,:] for d in _data]
Y = [np.r_[d, [tf.keras.utils.to_categorical(0,27)]][np.newaxis,:] for d in _data]

def test_data():
    assert data_size == 19909
    assert vocab_size == 27
    assert names[0] == 'aachenosaurus'
    assert ''.join([ix_to_char[x] for x in np.argmax(_data[0], axis=1)]) == 'aachenosaurus'


def test_ix_to_char():
    assert ix_to_char == {0: '\n', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

def test_SimpleRNN1():
    Wax = np.array([[3],[5]])
    Waa = np.array([[1,0],[0,1]])
    b = np.array([[0],[0]])

    x = np.array([1]).reshape(1,1)
    a_prev=np.zeros((2,1))

    assert np.allclose(Wax, [[3],[5]])
    assert np.allclose(x, [[1]])
    assert np.allclose(np.dot(Wax, x), [[3],[5]])
    assert np.allclose(Waa, [[1,0],[0,1]])
    assert np.allclose(b, [0,0])

    a = np.dot(Waa, a_prev)+np.dot(Wax, x)+b
    assert np.allclose(a, [[3],[5]])

    last = l0 = tf.keras.layers.Input(batch_shape=(1,1,1))
    last = l1 = tf.keras.layers.SimpleRNN(2, return_sequences=True, stateful=True, activation=tf.keras.activations.linear)(last)
    model = tf.keras.Model([l0], last)
    model.set_weights((Wax.T, Waa.T, b.T[0]))
    y = model.predict(x.reshape(1,1,1))
    assert np.allclose(y, [[[3,5]]])
    assert np.allclose(y, [a.T])

def test_SimpleRNN2():
    Wax = np.array([2,3,5,7]).reshape(2,2)
    assert np.allclose(Wax, [[2,3],[5,7]])
    Waa = np.array([[0,0],[0,0]])
    b = np.array([[0.3],[0.7]])

    x = np.array([11, 13]).reshape(2,1)
    assert np.allclose(x, [[11],[13]])
    a_prev=np.zeros((2,1))

    a = np.dot(Waa, a_prev)+np.dot(Wax, x)+b
    assert np.allclose(a, [[2*11+3*13+0.3],[5*11+7*13+0.7]])

    last = l0 = tf.keras.layers.Input(batch_shape=(1,1,2))
    last = l1 = tf.keras.layers.SimpleRNN(2, return_sequences=True, stateful=True, activation=tf.keras.activations.linear)(last)
    model = tf.keras.Model([l0], last)
    model.set_weights((Wax.T, Waa.T, b.T[0]))
    y = model.predict(x.reshape(1,1,2))
    assert np.allclose(y, [a.T])

def test_SimpleRNN3():
    Wax = np.array([2,3,5,7]).reshape(2,2)
    Waa = np.array([[17,19],[23,27]])
    b = np.array([[0.3],[0.7]])

    x = np.array([11, 13]).reshape(2,1)
    a_prev=np.zeros((2,1))

    a = np.dot(Waa, a_prev)+np.dot(Wax, x)+b
    a_prev = a
    a = np.dot(Waa, a_prev)+np.dot(Wax, x)+b

    last = l0 = tf.keras.layers.Input(batch_shape=(1,1,2))
    last = l1 = tf.keras.layers.SimpleRNN(2, return_sequences=True, stateful=True, activation=tf.keras.activations.linear)(last)
    model = tf.keras.Model([l0], last)
    model.set_weights((Wax.T, Waa.T, b.T[0]))
    y = model.predict(x.reshape(1,1,2))
    y = model.predict(x.reshape(1,1,2))
    assert np.allclose(y, [a.T])

def test_rnn_step_forward():
    with open('coursera/dino/testdata/test_sample_parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)
    with open('coursera/dino/testdata/test_rnn_step_forward.pkl', 'rb') as f:
        expected = pickle.load(f)
    assert parameters['b'].shape == (50,1)
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    b = parameters['b']
    Wya = parameters['Wya']
    by = parameters['by']
    model, getStep = mkModel(units=50)
    model.set_weights((Wax.T, Waa.T, b.T[0], Wya.T, by.T[0]))
    x = np.zeros((1,1,27))
    step = getStep()
    y = step.predict(x)
    e = expected['x'].reshape(1,1,27)
    assert np.allclose(y, e)

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
    assert np.allclose(loss3, loss)

def test_loss_gradients():
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
    assert np.allclose(loss3, loss)

    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5.0)

    b4 = [i.copy() for i in model.get_weights()]
    model.fit(X2,Y2, batch_size=1)
    after = [i.copy() for i in model.get_weights()]
    grads = [i-j for i,j in zip(after,b4)]
    g1 = grads[0].reshape(2700)
    g2 = gradients['dWax'].T.reshape(2700) * -0.01
    assert np.sum((g1-g2)**2)**0.5/np.sum(g1) < 1e-4
    assert np.sum((g1-g2)**2)**0.5 < 1e-5
