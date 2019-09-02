from block_sampler import *
import numpy as np

def test_files_from():
    result = files_from('../carving-experiments-19')
    assert '../carving-experiments-19/README.md' in result

def test_xs_encoder_one_hot():
    block = np.ones((512,)) * 255
    result = xs_encoder_one_hot([block])
    assert np.sum(result) == 512
    assert result[0,0,0] == 0
    assert result[0,0,255] == 1
    assert result.shape == (1,512,256)

def test_xs_encoder_8bits():
    block = np.ones((512,)) * 255
    result = xs_encoder_one_hot([block])
    assert np.sum(result) == 512
    assert result[0,0,0] == 0
    assert result[0,0,255] == 1
    assert result.shape == (1,512,256)

def test_mk_ys_encoder():
    ys_encoder = mk_ys_encoder(['a', 'b', 'c'])
    ys = ys_encoder(['a', 'a', 'a', 'c'])
    assert np.sum(ys) == 4
    assert ys[0,0] == 1
    assert ys[1,0] == 1
    assert ys[2,0] == 1
    assert ys[3,2] == 1
    ys_encoder = mk_ys_encoder(['c', 'b', 'a'])
    ys = ys_encoder(['a', 'a', 'a', 'c'])
    assert np.sum(ys) == 4
    assert ys[0,2] == 1
    assert ys[1,2] == 1
    assert ys[2,2] == 1
    assert ys[3,0] == 1
