from carving import *

def test_train():
    train('carving/dataset',1,10)

def test_ys_from_filenames():
    ys = ys_from_filenames(['a/a.pdf', 'b/b.png'])
    assert np.allclose(ys, [[1,0,0],[0,1,0]])

