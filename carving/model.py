import tensorflow as tf

def get_model(load_weights='model.h5'):
    last = l0 = tf.keras.layers.Input(shape=(512,256))
    last = tf.keras.layers.Conv1D(16, (4,), padding="same", activation="relu")(last)
    last = tf.keras.layers.MaxPooling1D(pool_size=(2,))(last)
    last = tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.5)(last)
    last = tf.keras.layers.LSTM(16, return_sequences=False, dropout=0.5)(last)
    last = tf.keras.layers.Dense(3)(last)
    last = tf.keras.layers.Activation('softmax')(last)

    model = tf.keras.Model([l0], last)

    if load_weights:
        try:
            model.load_weights(load_weights)
        except OSError:
            pass
    return model