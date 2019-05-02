import sys
import tensorflow as tf
from model import get_model
from sample import loadFolder
import numpy as np

def train(path, sample_size=10,batch_size=10):
    model = get_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(lr=0.1),
        metrics=['accuracy'])
    model.summary()

    while(True):
        xs, ys = loadFolder(path, sample_size=sample_size)
        model.fit(xs,ys,batch_size=batch_size)
        model.save('model.h5')
        y_pred = model.predict(xs[:5])
        for i, y_true in enumerate(ys[:5]):
            print(np.argmax(y_true), '\t', np.argmax(y_pred[i]))

if __name__ == '__main__':
    train(*sys.argv[1:])
