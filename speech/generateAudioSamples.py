from gtts import gTTS 
from pydub import AudioSegment
import numpy as np

def generateAudioSamples(text, lang='pt-BR'):
    sound = getAudioFeatures(text, lang)
    samples = normalize(sound.get_array_of_samples())
    return samples

def getAudioFeatures(text, lang='pt-BR'):
  myobj = gTTS(text=text, lang=lang, slow=False)
  myobj.save("temp.mp3")
  return AudioSegment.from_mp3("temp.mp3")

def normalize(samples):
  return samples/np.linalg.norm(samples, ord=np.inf)
