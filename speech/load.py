import numpy as np
from pydub import AudioSegment
import os, sys
from ctc_loss import chars_to_ix

def loadWav(path):
  segment = AudioSegment.from_wav(path)
  samples = segment.get_array_of_samples()
  ns = samples/np.linalg.norm(samples, ord=np.inf)
  return ns

def loadFolder(path):
  files = os.listdir(path)
  xs = []
  ys = []
  for f in files:
    y = f.split('.',1)[0]
    y = [chars_to_ix[j] for j in y]
    x = loadWav(os.path.join(path, f))
    xs.append(x)
    ys.append(y)
  return xs, ys

def to_ctc_format(xs,ys):
  max_tx = np.max([len(i) for in in xs])
  max_ty = np.max([len(i) for in in ys])
  xarr = np.zeros((len(xs), max_tx, 1))
  yarr = np.zeros((len(ys), max_ty + 2))
  for i, x in enumerate(xs):
    xarr[i,:len(x),0] = x
  for i, y in enumerate(ys):
    yarr[i,:len(y)] = [len(x), len(word), *y]
  return xarr, yarr

if __name__ == '__main__':
  xs, ys = loadFolder(sys.argv[1])
  print(ys)
