import utils.load
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

def get_accuracy(y_true, y_pred):
    matches = [i == j for i, j in zip(y_pred, y_true)]
    accuracy = sum(matches)/len(matches)
    return accuracy

class PrintAccuracyCB(Callback):
    def __init__(self, main):
        self.main = main
    def on_epoch_end(self, epoch, logs):
        if epoch%100 !=1:
            return
        xs, ys = self.main.data
        y_true, y_pred = self.main.get_true_pred(xs, ys)
        accuracy = get_accuracy(y_true, y_pred)
        print(' - accuracy: %0.2f' % accuracy)
        if accuracy >= self.main.goal_accuracy:
            self.main.model.stop_training = True

class AbstractMain(ABC):
    def __init__(self, command, save_file, examples_folder, batch_size, max_ty=100, sample_size=5, epochs=10000):
        self.save_file = save_file
        self.examples_folder = examples_folder
        self.batch_size = int(batch_size)
        self.max_ty = int(max_ty)
        self.sample_size = int(sample_size)
        self.epochs = int(epochs)
        self.callbacks = [
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
                log_dir=self.save_file.rsplit('.',1)[0] + '.tboard',
            ),
            PrintAccuracyCB(self),
        ]

        self.model = self.get_model()
        self.compile()
        utils.load.maybe_load_weigths(self.model, save_file=self.save_file)

        if command == 'train':
            self.train()
        elif command == 'evaluate':
            self.evaluate()
        elif command == 'predict':
            self.predict()
        elif command == 'correct':
            self.correct()
        elif command == 'failed':
            self.failed()
        else:
            print('Arguments command, save_file, examples_folder, batch_size=140, max_ty=100, sample_size=5, self.epochs=-1')
            print('commands: train, evaluate, predict, correct, failed')
            exit(1)

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def get_true_pred(self):
        pass

    @abstractmethod
    def get_xs_ys(self):
        pass

    def train(self):
        xs, ys = self.get_xs_ys()
        self.data = xs, ys
        accuracy = 0
        self.model.fit(xs,ys,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks)

    def evaluate(self):
        xs, ys = self.get_xs_ys()
        y_true, y_pred = self.get_true_pred(xs, ys)
        accuracy = get_accuracy(y_true, y_pred)
        print('accuracy: %0.2f' % accuracy)
        for i,j in zip(y_true, y_pred):
            print(i,j)

    def predict(self):
        xs, ys = self.get_xs_ys()
        y_pred = self.model.predict(xs,batch_size=self.batch_size)
        print(y_pred)

    def correct(self):
        xs, ys = self.get_xs_ys()
        y_true, y_pred = self.get_true_pred(xs, ys)
        for t,p in zip(y_true, y_pred):
            if t == p:
                print(repr(t), repr(p))

    def failed(self):
        xs, ys = self.get_xs_ys()
        y_true, y_pred = self.get_true_pred(xs, ys)
        for t,p in zip(y_true, y_pred):
            if t != p:
                print(repr(t), repr(p))
