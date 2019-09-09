import numpy as np
from train import *

def test_confusion():
    class A:
        pass
    def predict(xs):
        return np.array([0,1,0,1]).reshape((2,2))
    model = A()
    model.predict = predict
    xs = np.array([1,2,3,4]).reshape((2,2))
    ys = np.array([0,1,1,0]).reshape((2,2))
    results = confusion(model, xs, ys, {0:'0', 1: '1'})
    assert results[0] == {"True": '0', '0': 1, '1': 0}
    