from block_sampler import *
import numpy as np

def test_files_from():
    result = files_from('../carving-experiments-19')
    assert '../carving-experiments-19/README.md' in result

def test_xs_encoder_one_hot():
    block = np.ones((512,), dtype='int') * 255
    result = xs_encoder_one_hot([block])
    assert np.sum(result) == 512
    assert result[0,0,0] == 0
    assert result[0,0,255] == 1
    assert result.shape == (1,512,256)

def test_xs_encoder_8bits01():
    block = np.ones((512,), dtype='int') * 17
    result = xs_encoder_8bits01([block])
    assert np.sum(result) == 1024
    assert (result[0,0] == [0,0,0,1,0,0,0,1]).all()
    assert result.shape == (1,512,8)

def test_xs_encoder_8bits_11():
    block = np.ones((512,), dtype='int') * 17
    result = xs_encoder_8bits_11([block])
    assert np.sum(result) == -2048
    assert (result[0,0] == [-1,-1,-1,1,-1,-1,-1,1]).all()
    assert result.shape == (1,512,8)

def test_xs_encoder_16bits():
    block = np.ones((512,), dtype='int') * 17
    result = xs_encoder_16bits([block])
    assert (result[0,0] == [0,0,0,1,0,0,0,1, 1,1,1,0,1,1,1,0]).all()
    assert np.sum(result) == 4096
    assert result.shape == (1,512,16)

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
