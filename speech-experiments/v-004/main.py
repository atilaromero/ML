import tensorflow as tf
import sys
sys.path.append("../..")
import utils.load

from utils.abstract_main import AbstractMain, AccuracyCB
from ctc.ctc_loss import chars_to_ix, to_ctc_format, ctc_loss, ctc_predict, from_ctc_format

print("tf.VERSION", tf.VERSION)
print("tf.keras.__version__", tf.keras.__version__)

from audio.fft import spectrogram_from_file

def xs_ys_from_filenames(filenames, max_ty):
    xs = []
    ys = []
    for f in filenames:
        y = utils.load.category_from_name(f)
        y = [chars_to_ix[j] for j in y]
        x = spectrogram_from_file(f)
        xs.append(x)
        ys.append(y)
    xs, ys = to_ctc_format(xs, ys, max_ty)
    return xs, ys

class Custom(AbstractMain):
    def get_callbacks(self):
        return [
            # tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ModelCheckpoint(
                self.save_file, 
                monitor='loss',
                save_best_only=True,
                save_weights_only=True,
                period=100),
            # tf.keras.callbacks.ReduceLROnPlateau(
            #     monitor='loss',
            #     patience=10,
            #     factor=0.8,
            #     min_lr=1e-6,
            # ),
            # tf.keras.callbacks.EarlyStopping(
            #     monitor='loss',
            #     min_delta=0.001,
            #     patience=500,
            # ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.save_file.rsplit('.',1)[0] + '.tboard'
            ),
            AccuracyCB(self, show=True, stop=1.0, interval=500),
        ]

    def get_model(self):
        last = l0 = tf.keras.layers.Input(shape=(None,221))
        last = tf.keras.layers.Conv1D(16, (3,), padding="same", activation="relu")(last)
        last = tf.keras.layers.MaxPool1D()(last)
        last = tf.keras.layers.Conv1D(8, (3,), padding="same", activation="relu")(last)
        last = tf.keras.layers.MaxPool1D()(last)
        last = tf.keras.layers.Conv1D(8, (3,), padding="same", activation="relu")(last)
        last = tf.keras.layers.MaxPool1D()(last)
        last = tf.keras.layers.Conv1D(8, (3,), padding="same", activation="relu")(last)
        last = tf.keras.layers.MaxPool1D()(last)
        last = tf.keras.layers.Conv1D(8, (3,), padding="same", activation="relu")(last)
        last = tf.keras.layers.MaxPool1D()(last)
        last = tf.keras.layers.Conv1D(8, (3,), padding="same", activation="relu")(last)
        last = tf.keras.layers.MaxPool1D()(last)
        last = tf.keras.layers.Conv1D(4, (3,), padding="same", activation="relu")(last)
        last = tf.keras.layers.LSTM(64, return_sequences=True)(last)
        last = tf.keras.layers.Dense(27)(last)
        last = tf.keras.layers.Activation('softmax')(last)

        model = tf.keras.Model([l0], last)
        return model

    def get_true_pred(self, xs, ys):
        y_pred = ctc_predict(self.model, xs)
        y_true = from_ctc_format(ys)
        return y_true, y_pred

    def compile(self):
        self.model.compile(loss=ctc_loss((self.batch_size, self.max_ty)),
            optimizer=tf.keras.optimizers.Adam())

    def get_xs_ys(self):
        examples = utils.load.examples_from(self.examples_folder)
        sample = examples
        xs, ys = xs_ys_from_filenames(sample, self.max_ty)
        return xs, ys

if __name__ == '__main__':
    import time
    start = time.time()
    if len(sys.argv) == 1:
        Custom('train', 'v.h5', '../../datasets/speech/syllables/v/', 5)
    else:
        Custom(*sys.argv[1:])
    print('elapsed time:', time.time() - start)
