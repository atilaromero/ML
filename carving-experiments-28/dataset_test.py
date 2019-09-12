from dataset import Dataset

def test_dataset():
    d = Dataset(['dataset.py'])
    assert d.ix_to_cat[0] == 'py'
    assert d.categories == ['py']
    d = d.join(Dataset(['a.b', 'c.d']))
    assert d.categories == ['b', 'd', 'py']
    d = Dataset(['dataset.py'])
    for b in d.generator():
        assert b.category in ['py']
        break
    for b in d.generator(distribution='by_sector'):
        assert b.category in ['py']
        break
