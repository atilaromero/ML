from dataset import Dataset
from block_sampler import BlockSamplerByFile, BlockSamplerBySector

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
