import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Activation, TimeDistributed, Flatten, Dot, Softmax, Lambda
import tensorflow.keras.backend as K

from run_experiments import Experiment, run_experiments, save_experiment_results, sector_generator
from main import CCCAD3

sys.path.append('..')
import utils
import utils.load
import utils.sampler

model = tf.keras.models.load_model('CCCAD3.h5')
model.summary()

train = utils.load.examples_from('../datasets/carving/train/pdf')[:1]
gen = sector_generator(train, 1, 'first')
data = next(gen)

layer_name = 'softmax'
intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)

def hexdump(chars):
    FILTER = ''.join([(len(repr(chr(x))) == 3) and chr(x) or '.' for x in range(256)])
    hex = ["%02x" % x for x in chars]
    printable = ["%s" % ((x <= 127 and FILTER[x]) or '.') for x in chars]
    return hex, printable

import numpy as np
xmax=np.max(intermediate_output)
xmin=np.min(intermediate_output)
x = np.array(intermediate_output).reshape((512))
x=(x-xmin)/(xmax-xmin)
x = 255-(250*x+5)

print(model.predict(data))
data2=np.argmax(data[0][0],axis=1)
print(data2.shape)

hex, printable = hexdump(data2)
with open('hex.html', 'w') as f:
    f.write('<div style="font-family:monospace;">')
    for i in range(0,512,16):
        for j in range(16):
            f.write('<span style="background-color: rgba(255,%d,%d,1);">%s </span>'%(x[i+j],x[i+j],hex[i+j]))
        f.write(" | ")
        for j in range(16):
            f.write('<span style="background-color: rgba(255,%d,%d,1);">%s</span>'%(x[i+j],x[i+j],printable[i+j]))
        f.write("<br/>")
    f.write('</div>')

