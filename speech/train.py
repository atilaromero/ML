
import numpy as np
from load import loadFolder
from ctc_loss import to_ctc_format
from model import get_model

xs, ys = loadFolder('dataset')
xarr, yarr = to_ctc_format(xs, ys)
assert len(xarr) == len(yarr)

sample_size = 10
model = get_model((sample_size, yarr.shape[1]))

while(True):
    idx = np.random.choice(np.arange(len(xarr)), sample_size, replace=False)
    model.fit(xarr[idx],yarr[idx])
    model.save('model.h5')
