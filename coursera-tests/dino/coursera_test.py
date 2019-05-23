import numpy as np
from utils import *
import random
from coursera import *
import pickle
import os 

dinopath = os.path.join(os.path.dirname(__file__), 'dinos.txt')

data = open(dinopath, 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

def test_data():
    assert data_size == 19909
    assert vocab_size == 27

def test_ix_to_char():
    assert ix_to_char == {0: '\n', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

def test_clip():
    with open('coursera/dino/testdata/test_clip_gradients.pkl', 'rb') as f:
        gradients = pickle.load(f)
    gradients = clip(gradients, 10)
    assert gradients["dWaa"][1][2] == 10.0
    assert gradients["dWax"][3][1] ==-10.0
    assert gradients["dWya"][1][2] == 0.2971381536101662
    assert all(gradients["db"][4] == [10.0])
    assert all(gradients["dby"][1] == [8.45833407057182])

def test_sample():
    with open('coursera/dino/testdata/test_sample_parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)

    indices = sample(parameters, char_to_ix, 0)
    assert all([i==j for i, j in zip(indices, [12, 26, 22, 20, 15, 25, 20, 1, 14, 14, 14, 14, 8, 3, 14, 5, 14, 2, 7, 3, 2, 3, 15, 15, 17, 26, 26, 5, 23, 14, 15, 3, 12, 13, 15, 14, 4, 20, 22, 10, 15, 20, 22, 12, 15, 20, 11, 21, 12, 25, 0])])
    assert all([ix_to_char[i] == j for i, j  in zip(indices, ['l', 'z', 'v', 't', 'o', 'y', 't', 'a', 'n', 'n', 'n', 'n', 'h', 'c', 'n', 'e', 'n', 'b', 'g', 'c', 'b', 'c', 'o', 'o', 'q', 'z', 'z', 'e', 'w', 'n', 'o', 'c', 'l', 'm', 'o', 'n', 'd', 't', 'v', 'j', 'o', 't', 'v', 'l', 'o', 't', 'k', 'u', 'l', 'y', '\n'])])

def test_rnn_step_forward():
    with open('coursera/dino/testdata/test_sample_parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)
    with open('coursera/dino/testdata/test_rnn_step_forward.pkl', 'rb') as f:
        expected = pickle.load(f)
    x = np.zeros((27,1))
    a = np.zeros(parameters['b'].shape)
    a, x = rnn_step_forward(parameters,a, x)
    assert np.sum((x - expected["x"])**2) < 1e-16
    assert np.sum((a - expected["a"])**2) < 1e-16

def test_optimize():
    np.random.seed(1)
    vocab_size, n_a = 27, 100
    a_prev = np.random.randn(n_a, 1)
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    X = [12,3,5,11,22,3]
    Y = [4,14,11,22,25, 26]

    loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
    assert np.allclose(loss, 126.50397572165365)
    assert np.allclose(gradients["dWaa"][1][2],0.19470931534721261)
    assert np.allclose(np.argmax(gradients["dWax"]),93)
    assert np.allclose(gradients["dWya"][1][2],-0.0077738760320040928)
    assert np.allclose(gradients["db"][4].tolist(),[-0.068098250152480944])
    assert np.allclose(gradients["dby"][1].tolist(), [0.015381922316513334])
    assert np.allclose(a_last[4],[-1.])
