from dataset import Dataset
from block_sampler import BlockSamplerByFile, BlockSamplerBySector, RandomSampler, BlockSamplerByCategory


def test_block_sampler():
    d = Dataset(['dataset.py'])
    bs = BlockSamplerByFile(d)
    for b in bs:
        assert b.category in ['py']
        break
    bs = BlockSamplerBySector(d)
    for b in bs:
        assert b.category in ['py']
        break
    bs = BlockSamplerByCategory(d)
    for b in bs:
        assert b.category in ['py']
        break
    rs = RandomSampler(bs)
    for b in rs:
        assert b.category in ['random', 'not_random']
        break
