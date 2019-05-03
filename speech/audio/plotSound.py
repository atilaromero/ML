import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

def plotSound(sound, path='plot.png'):
  if type(sound) == AudioSegment:
    y = sound.get_array_of_samples()
  else:
    y = sound #samples
  t = np.arange(0,len(y))
  plt.plot(t, y)
  plt.savefig(path)
  plt.close()

