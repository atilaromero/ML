import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

with open("dinos.txt") as f:
  names = f.readlines()
names = [x.lower().strip() for x in names]

char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

def lines2array(lines, char_to_ix, ix_to_char):
  lenchars = len(char_to_ix)
  result = []
  for i, line in enumerate(lines):
    result.append(np.zeros((len(line),lenchars)))
    for j, c in enumerate(line):
      result[i][j]= keras.utils.to_categorical(char_to_ix[c], lenchars)
  return result

def mkModel(units=50):
  last = l0 = keras.layers.Input(shape=(None,27))
  last = l1 = keras.layers.SimpleRNN(units, return_sequences=True, activation='tanh')(last)
  last = l2 = keras.layers.Dense(27, activation='softmax')(last)
  model=keras.Model([l0], last)
  last = l0 = keras.layers.Input(batch_shape=(1,1,27))
  last = l1 = keras.layers.SimpleRNN(units, return_sequences=True, activation='tanh', stateful=True)(last)
  last = l2 = keras.layers.Dense(27, activation='softmax')(last)
  step=keras.Model([l0], last)
  def getStep():
    step.set_weights(model.get_weights())
    return step
  return model, getStep

def sample(model):
  model.reset_states()
  letter=0
  result=[]
  for i in range(50):
    item = []
    chances = model.predict(np.array([keras.utils.to_categorical([letter],27)]))[0]
    letter  = np.random.choice(range(27), p=chances.ravel())
    result.append(ix_to_char[letter])
  return ''.join(result).split('\n')[0]