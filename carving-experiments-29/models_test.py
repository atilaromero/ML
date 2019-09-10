import models

def test_args_gen():
    layers = list(models.args_gen([1],[2],a=[3], b=[4]))
    assert str(layers) == "[([1, 2], {'a': 3, 'b': 4})]"
    layers = list(models.args_gen([1],[2,20],a=[3], b=[4]))
    assert str(layers) == "[([1, 2], {'a': 3, 'b': 4}), ([1, 20], {'a': 3, 'b': 4})]"
    layers = list(models.args_gen([1],[2],a=[3], b=[4,5]))
    assert str(layers) == "[([1, 2], {'a': 3, 'b': 4}), ([1, 2], {'a': 3, 'b': 5})]"
    layers = list(models.args_gen([1],[2],a=[3], b=[4,None]))
    assert str(layers) == "[([1, 2], {'a': 3, 'b': 4}), ([1, 2], {'a': 3})]"
    layers = list(models.args_gen([1],[2, None],a=[3], b=[4]))
    assert str(layers) == "[([1, 2], {'a': 3, 'b': 4}), ([1], {'a': 3, 'b': 4})]"

if __name__ == '__main__':
    test_args_gen()