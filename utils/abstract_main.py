import utils.load
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_accuracy(y_true, y_pred):
    matches = [i == j for i, j in zip(y_pred, y_true)]
    accuracy = sum(matches)/len(matches)
    return accuracy


class AccuracyCB(Callback):
    def __init__(self, main, show=True, stop=None, interval=10):
        self.main = main
        self.show = show
        self.stop = stop
        self.interval = interval
        self.loss_goal = 0.5
    def should_increase_dataset(self, loss):
        return hasattr(self.main, 'data_step') and loss < self.loss_goal
    def should_stop_training(self, accuracy):
        if hasattr(self.main, 'alldata'):
            if len(self.main.data[0]) < len(self.main.alldata[0]):
                return False
        return self.stop and accuracy >= self.stop
    def increase_dataset(self):
        xs, ys = self.main.data
        axs, ays = self.main.alldata
        if len(xs) <= len(axs):
            xs = axs[:len(xs)+1]
            ys = ays[:len(ys)+1]
            self.main.data = xs, ys
    def on_epoch_end(self, epoch, logs):
        if self.should_increase_dataset(logs['loss']):
            self.increase_dataset()
        if epoch % self.interval != 0:
            return
        y_true, y_pred = self.main.get_true_pred(*self.main.data)
        accuracy = get_accuracy(y_true, y_pred)
        if self.show:
            for t,p in zip(y_true, y_pred):
                print(repr(t), repr(p))
            print(' - accuracy: %0.2f' % accuracy)
        if self.should_stop_training(accuracy):
            self.model.stop_training = True

class AbstractMain(ABC):
    def __init__(self, command, save_file, examples_folder, batch_size, max_ty=100, sample_size=5, epochs=1000000):
        self.save_file = save_file
        self.examples_folder = examples_folder
        self.batch_size = int(batch_size)
        self.max_ty = int(max_ty)
        self.sample_size = int(sample_size)
        self.epochs = int(epochs)
        self.callbacks = self.get_callbacks()

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
        elif command == 'fit_generator':
            self.fit_generator()
        else:
            print('Arguments command, save_file, examples_folder, batch_size=140, max_ty=100, sample_size=5, self.epochs=-1')
            print('commands: train, evaluate, predict, correct, failed, fit_generator')
            exit(1)

    @abstractmethod
    def get_callbacks(self):
        pass

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
        history = self.model.fit(xs,ys,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks)
        keys = history.history.keys()
        for k in keys:
            plt.plot(history.history[k])
        plt.legend(keys)
        plt.savefig('history.png')

    def fit_generator(self):
        xs, ys = self.get_xs_ys()
        #index of ys sorted by y[:,1], which is len(string),
        # to use shortest strings first
        p = np.argsort(ys[:,1])
        xs, ys = xs[p], ys[p]
        self.alldata = xs, ys
        self.data_step = 5
        self.data = xs[:self.data_step], ys[:self.data_step]
        def gen1():
            lenxs=0
            while(True):
                xs, ys = self.data
                p = np.random.choice(list(range(len(xs))), size=self.batch_size, replace=True)
                yield xs[p], ys[p]
        self.model.fit_generator(gen1(), 
            epochs=self.epochs,
            steps_per_epoch=1,
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
