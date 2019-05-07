import numpy as np
from pydub import AudioSegment
import os, sys
from ctc_loss import chars_to_ix
from fft import spectrogram_from_file

def loadWav(path):
  segment = AudioSegment.from_wav(path)
  samples = segment.get_array_of_samples()
  ns = samples/np.linalg.norm(samples, ord=np.inf)
  return ns[:,np.newaxis]

def loadFolder(path, sample_size=-1):
  files = os.listdir(path)
  xs = []
  ys = []
  if sample_size > 0:
    files = np.random.choice(files, size=sample_size, replace=True)
  for f in files:
    y = f.split('.',1)[0]
    y = [chars_to_ix[j] for j in y]
    # x = loadWav(os.path.join(path, f))
    x = spectrogram_from_file(os.path.join(path, f))
    xs.append(x)
    ys.append(y)
  return xs, ys

