from batch_encoder import *
import numpy as np
from block_sampler import BlockSamplerByFile

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
    ys_encoder = mk_ys_encoder({'a':0, 'b':1, 'c':2})
    ys = ys_encoder(['a', 'a', 'a', 'c'])
    assert np.sum(ys) == 4
    assert ys[0,0] == 1
    assert ys[1,0] == 1
    assert ys[2,0] == 1
    assert ys[3,2] == 1
    ys_encoder = mk_ys_encoder({'a':2, 'b':1, 'c':0})
    ys = ys_encoder(['a', 'a', 'a', 'c'])
    assert np.sum(ys) == 4
    assert ys[0,2] == 1
    assert ys[1,2] == 1
    assert ys[2,2] == 1
    assert ys[3,0] == 1

from dataset import Dataset

def test_new():
    d = Dataset(['dataset.py'])
    bs = BlockSamplerByFile(d)
    be = BatchEncoder(bs, 10, 'one_hot')
    for xs, ys in be:
        assert len(xs) == 10
        assert len(ys) == 10
        assert xs.shape == (10, 512, 256)
        assert ys.shape == (10, 1)
        break