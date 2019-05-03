from gtts import gTTS 
from pydub import AudioSegment
import numpy as np
import tempfile
import os

def generateAudioSamples(text, lang='pt-BR'):
    sound = getAudioFeatures(text, lang)
    samples = normalize(sound.get_array_of_samples())
    return samples

def getAudioFeatures(text, lang='pt-BR'):
  myobj = gTTS(text=text, lang=lang, slow=False)
  f, name = tempfile.mkstemp()
  os.close(f)
  myobj.save(name)
  segment = AudioSegment.from_mp3(name)
  os.remove(name)
  return segment

def normalize(samples):
  return samples/np.linalg.norm(samples, ord=np.inf)
