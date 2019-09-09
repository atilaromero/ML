import os
import sys
import time
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow.keras.backend as K
import time

from queue import Queue
from threading import Thread

def main(h5model, rawfile, threshold=0.5):
    threshold = float(threshold)
    model = tf.keras.models.load_model(h5model)
    batch_size = 64
    i = 0
    last = time.time()

    blks = blocks(rawfile)
    chunks = parallel_iterable(batch(blks, batch_size))
    for chunk in chunks:
        chunk=np.array(chunk)
        results = model.predict(chunk)
        MBsec = 1024**2/512
        if i%(MBsec) == 0 and i>0:
            now = time.time()
            print("MB/s:", 1/(now-last), i*512/1024/1024, file=sys.stderr)
            last = now
        for j in range(results.shape[0]):
            if results[j,0] > threshold:
                print(i, results[j])
            i+=1

def parallel_iterable(iterable, maxsize=10):
    class EndQueue:
        pass
    q = Queue(maxsize=maxsize)
    def f(q, iterable):
        for x in iterable:
            q.put(x)
        q.put(EndQueue)
    worker = Thread(target=f, args=(q, iterable))
    worker.setDaemon(True)
    worker.start()
    while True:
        x = q.get()
        if x is EndQueue:
            return
        yield x

def blocks(rawfile):
    def myread(rawfile):
        with open(rawfile, 'rb') as f:
            while True:
                b = f.read(512)
                if len(b) == 0:
                    return
                yield b
    for b in parallel_iterable(myread(rawfile)):
        _bytes=[int(x) for x in b]
        _features = one_hot(_bytes, 256)
        n = np.zeros((512,256),dtype='int')
        n[:len(b)] = _features
        yield n

def batch(iterable, n):
    iterable=iter(iterable)
    while True:
        chunk=[]
        for i in range(n):
            try:
                chunk.append(next(iterable))
            except StopIteration:
                yield chunk
                return
        yield chunk

def one_hot(arr, num_categories):
    arr_shape = np.shape(arr)
    flatten = np.reshape(arr, -1)
    r = np.zeros((len(flatten),num_categories))
    r[np.arange(len(flatten)),flatten] = 1
    return r.reshape((*arr_shape,num_categories))

if __name__ == '__main__':
    main(*sys.argv[1:])