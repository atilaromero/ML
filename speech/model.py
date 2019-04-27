import tensorflow as tf

last = l0 = tf.keras.layers.Input(shape=(None,1))
# last = tf.keras.layers.Masking(mask_value=100)(last)
# last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
# last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
last = tf.keras.layers.Dense(27)(last)
last = tf.keras.layers.Activation('softmax')(last)

model = tf.keras.Model([l0], last)

model.compile(loss=ctc_loss(yarr.shape),
    optimizer=tf.keras.optimizers.SGD(lr=0.001))

try:
    model.load_weights('model.h5')
except OSError:
    pass

