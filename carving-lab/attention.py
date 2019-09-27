from IPython.core.display import HTML

import numpy as np
import matplotlib.pyplot as plt
from batch_encoder import decode_8bits_11

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Activation, TimeDistributed, Flatten, Dot, Softmax, Lambda
import tensorflow.keras.backend as K


def Attention(timesteps, nfeatures, name):
    def f(last):
        TIMESTEPS = timesteps
        NFEATURES = nfeatures
        query = last                             # (None, TIMESTEPS, NFEATURES)
        # pick one of the convolutions
        query = MaxPooling1D(pool_size=TIMESTEPS, strides=1)(
            query)  # (None, 1, NFEATURES)
        # remove dimension with size 1
        query = Lambda(lambda q: K.squeeze(q, 1))(query)    # (None, NFEATURES)
        query = Dense(NFEATURES)(query)                     # (None, NFEATURES)
        attScores = Dot(axes=[1, 2])([query, last])         # (None, TIMESTEPS)
        attScores = Softmax(name=name)(attScores)
        # apply attention scores
        attVector = Dot(axes=[1, 1])([attScores, last])     # (None, NFEATURES)
        last = attVector
        return last
    return f


def showHex(sample, model, attention_layer_model, n, decoder=decode_8bits_11):
    attention_output = attention_layer_model.predict(sample)
    data2 = decoder(sample[0][n])
    x = normColor(attention_output[n])
    hex, printable = hexdump(data2)
    s = ""
    s += '<div style="font-family:monospace;font-size:10px;line-height: 10px;">'
    for i in range(0, 512, 16):
        #         s+="<div stype='border: 0;'>"
        for j in range(16):
            s += '<span style="background-color: rgba(255,%d,%d,1);">%s </span>' % (
                x[i+j], x[i+j], hex[i+j])
        s += " | "
        for j in range(16):
            s += '<span style="background-color: rgba(255,%d,%d,1);">%s</span>' % (
                x[i+j], x[i+j], printable[i+j])
#         s+='<br style=\'display: block;content: "";margin: 0;border:0;\' />'
        s += '<br/>'
    s += ('</div>')
    return HTML(s)


def hexdump(chars):
    FILTER = ''.join([(len(repr(chr(x))) == 3) and chr(x)
                      or '.' for x in range(256)])
    hex = ["%02x" % int(x) for x in chars]
    printable = ["%s" % ((x <= 127 and FILTER[x]) or '.') for x in chars]
    return hex, printable


def normColor(attention_output):
    xmax = np.max(attention_output)
    xmin = np.min(attention_output)
    x = np.array(attention_output).reshape((512))
    x = (x-xmin)/(xmax-xmin)
    x = 255-(250*x+5)
    return x


def showSample(sample, model, attention_layer_model, ix_to_cat, n):
    prediction = model.predict(sample)
    print(np.argmax(prediction[n]))
    print(ix_to_cat[np.argmax(prediction[n])])
    plt.ylim((0, 1))
    plt.xticks(rotation=90)
    plt.bar([ix_to_cat[x] for x in range(28)], prediction[n])
    plt.bar([ix_to_cat[x]
             for x in [np.argmax(sample[1][n])]], prediction[n])
    plt.show()
    return showHex(sample, model, attention_layer_model, n)
