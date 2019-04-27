
from load import loadFolder
from ctc_loss import to_ctc_format

xs, ys = loadFolder('dataset')
xarr, yarr = to_ctc_format(xs, ys)
print(xarr.shape, yarr.shape)