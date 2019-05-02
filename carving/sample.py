import os
import sys
import numpy as np
import tensorflow as tf

categories = ['pdf','png', 'jpg']
ix_to_cat = dict([(i,x) for i,x in enumerate(categories)])
cat_to_ix = dict([(x,i) for i,x in enumerate(categories)])

def sample_sector(path):
    with open(path, 'rb') as f:
        f.seek(0,2)
        size = f.tell()
        i = np.random.randint(0,size)
        sector = i//512
        f.seek(sector*512,0)
        b = f.read(512)
        n = np.zeros((512),dtype='int')
        n[:len(b)] = [int(x) for x in b]
        return n

def loadFolder(path, sample_size=-1):
  files = os.listdir(path)
  sample_size = int(sample_size)
  if sample_size > 0:
    files = np.random.choice(files, size=sample_size, replace=True)
  xs = np.zeros((len(files),512,256))
  ys = np.zeros((len(files),len(categories)))
  for i,f in enumerate(files):
    y = cat_to_ix[f.rsplit('.',1)[-1]]
    x = sample_sector(os.path.join(path, f))
    # one hot encoding
    xs[i,np.arange(512),x] = 1
    ys[i,y] = 1
  return xs, ys

if __name__ == '__main__':
    xs, ys = loadFolder(*sys.argv[1:])
    print(np.argmax(xs,axis=2)[:,:3], np.argmax(ys,axis=-1))