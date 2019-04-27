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

