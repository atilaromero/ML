import tensorflow as tf
import numpy as np
from generateAudioSamples import generateAudioSamples
import tensorflow.keras.backend as K
import time
import sys

last = l0 = tf.keras.layers.Input(shape=(None,1))
# last = tf.keras.layers.Masking(mask_value=100)(last)
# last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
# last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.Dense(27)(last)
last = tf.keras.layers.Activation('softmax')(last)

def ctc_loss(y_shape):
  def f(y_true, y_pred):
    y_true = tf.reshape(y_true, y_shape)
    k_inputs = y_pred
    k_input_lens = y_true[:,0:1]
    k_label_lens = y_true[:,1:2]
    k_labels = y_true[:,2:]
    cost = K.ctc_batch_cost(k_labels, k_inputs, k_input_lens,k_label_lens)
    return cost
  return f


model = tf.keras.Model([l0], last)
# model.summary()

out_chars = 'abcdefghijklmnopqrstuvwxyz '
chars_to_ix = dict(zip(out_chars,range(len(out_chars))))
ix_to_chars = dict(zip(range(len(out_chars)),out_chars))
assert len(out_chars)==27
assert chars_to_ix['b'] == 1
assert ix_to_chars[2] == 'c'

def generate_syllables(n=-1):
  while(n!=0):
    c = np.random.choice(['','b','c','d','f','g','j','k','l','m','n','p','q','r','s','t','v','x','z'])
    v = np.random.choice(['a','e','i','o','u'])
    yield c + v
    n-=1

print("creating samples")
t = time.time()
xs = [] # array of instances
ys = []
max_x = 0 # max of len(x)
max_y = 0
for i, word in enumerate(generate_syllables(100)):
    print('.',end='')
    sys.stdout.flush()
    # audio amplitudes
    x = generateAudioSamples(word)
    # first two numbers are parameters for ctc_loss function
    y = [len(x), len(word), *[chars_to_ix[j] for j in word]]
    xs.append(x)
    ys.append(y)
    max_x = max(max_x, len(x))
    max_y = max(max_y, len(y))
print()

print(time.time() - t)
t = time.time()
print("creating matrices")
# xs has variable size, but must be converted to a np.array
xarr = np.zeros((len(xs), max_x,1))
yarr = np.zeros((len(ys), max_y))
for i,x in enumerate(xs):
    xarr[i,:len(x),0] = x
for i,y in enumerate(ys):
    yarr[i,:len(y)] = y

try:
    model.load_weights('model.h5')
except OSError:
    pass

print(time.time() - t)
t = time.time()
print("compiling model")
model.compile(loss=ctc_loss(yarr.shape),
    optimizer=tf.keras.optimizers.SGD(lr=0.001))

for i in range(10):
    model.fit(xarr, yarr,epochs=10)
    model.save_weights("model.h5")
