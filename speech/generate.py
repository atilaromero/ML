import numpy as np
import subprocess
import sys
from pydub import AudioSegment

consonants = ['','b', 'br','c', 'cr','d', 'dr','f', 'fr','g', 'gr','j','l', 'lh','m','n', 'nh','p','pr','qu','r','s','t', 'tr','v','vr','x','z']
vowels = ['a','e','i','o','u']
suffixes = ['', 's', 'r', 'l', 'm']
def generate_syllables(n=-1):
  while(n!=0):
    yield ''.join([np.random.choice(x) for x in [consonants,vowels,suffixes]])
    n-=1

def generateAllSyllables():
  for c in consonants:
    for v in vowels:
        for s in suffixes:
            yield c+v+s

def saveSound(word):
  p = subprocess.Popen(['espeak-ng','-v', 'pt-BR', word, '--stdout'], bufsize=-1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = p.communicate()
  with open(word+'.wav', 'wb') as f:
    f.write(out)

if __name__ == '__main__':
  for x in generateAllSyllables():
    saveSound(x)  
