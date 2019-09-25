from dataset import Dataset

def test_dataset():
    d = Dataset(['dataset.py'])
    assert d.ix_to_cat[0] == 'py'
    assert d.categories == ['py']
    d = d.join(Dataset(['a.b', 'c.d']))
    assert d.categories == ['b', 'd', 'py']
    d = d.join(Dataset([
                        '1.b','1.d','1.py',
                        '1.b','1.d','1.py',
                        '1.b','1.d','1.py',
                        '1.b','1.d','1.py',
                        '1.b','1.d','1.py',
                        '1.b','1.d','1.py',
                        '1.b','1.d','1.py',
                        '2.b','2.d','2.py',
                        '3.b','3.d','3.py',
                        '4.b','4.d','4.py',
                        '5.b','5.d','5.py',
                        '6.b','6.d','6.py',
                        '7.b','7.d','7.py',
                        '8.b','8.d','8.py',
                        '9.b','9.d','9.py',
                        '10.b','10.d','10.py',
                        '11.b','11.d','11.py',
                        '12.b','12.d','12.py',
                        '13.b','13.d','13.py',
                        '14.b','14.d','14.py',
                        '15.b','15.d','15.py',
                        '16.b','16.d','16.py',
                        '17.b','17.d','17.py',
                        '18.b','18.d','18.py',
                        '19.b','19.d','19.py',
                        ]))
    assert len(d.filenames) == 3*20
    x = d.rnd_split_num_by_category(3)
    y = next(iter(x))
    assert len(y.filenames) == 9
    for k, v in y.by_category().items():
        assert len(v) == 3
    x = d.rnd_split_fraction_by_category(0.5)
    y = next(iter(x))
    assert len(y.filenames) == 30
    for k, v in y.by_category().items():
        assert len(v) == 10
