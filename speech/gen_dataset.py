import numpy as np
import subprocess
import os
import sys
from pydub import AudioSegment

consonants = ['','b', 'br','c', 'cr','d', 'dr','f', 'fr','g', 'gr','j','l', 'lh','m','n', 'nh','p','pr','qu','r','s','t', 'tr','v','vr','x','z']
vowels = ['a','e','i','o','u']
suffixes = ['', 's', 'r', 'l', 'm']
def generate_syllables(n=-1):
  while(n!=0):
    yield ''.join([np.random.choice(x) for x in [consonants,vowels,suffixes]])
    n-=1

def generateAllSyllables(pattern='cvs'):
  for c in ('c' in pattern) and consonants or ['']:
    for v in ('v' in pattern) and vowels or ['']:
      for s in ('s' in pattern) and suffixes or ['']:
        yield c+v+s

def saveSound(word, dir):
  p = subprocess.Popen(['espeak','-v', 'pt-br', word, '--stdout'], bufsize=-1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, _ = p.communicate()
  with open(os.path.join(dir, word)+'.wav', 'wb') as f:
    f.write(out)

if __name__ == '__main__':
  os.makedirs(sys.argv[1],exist_ok=True)
  for x in generateAllSyllables(*sys.argv[2:]):
    saveSound(x, sys.argv[1])  
