import numpy as np

def test_one_hot():
    assert np.allclose(one_hot([0,1,2,1,0],3), [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]])
    assert np.allclose(one_hot([[0,1,2,1,0]],3), [[[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]]])

