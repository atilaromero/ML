import tensorflow as tf
from ctc_loss import ctc_loss

def get_model(load_weights='model.h5'):
    last = l0 = tf.keras.layers.Input(shape=(None,221))
    # last = tf.keras.layers.Masking(mask_value=100)(last)
    # last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
    # last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.Dense(27)(last)
    last = tf.keras.layers.Activation('softmax')(last)

    model = tf.keras.Model([l0], last)

    if load_weights:
        try:
            model.load_weights(load_weights)
        except OSError:
            pass
    return model
