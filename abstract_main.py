import utils.load
import numpy as np
from abc import ABC, abstractmethod
from tensorflow.keras.callbacks import Callback

from ctc.ctc_loss import ctc_predict, from_ctc_format

def get_accuracy(y_true, y_pred):
    matches = [i == j for i, j in zip(y_pred, y_true)]
    accuracy = sum(matches)/len(matches)
    return accuracy

class print_accuracy_cb(Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = self.get_true_pred(xs, ys)
        accuracy = get_accuracy(y_true, y_pred)
        print('accuracy: %0.2f' % accuracy)

class AbstractMain(ABC):
    def __init__(self, command, save_file, examples_folder, batch_size, max_ty=100, sample_size=5, epochs=-1):
        self.save_file = save_file
        self.examples_folder = examples_folder
        self.batch_size = int(batch_size)
        self.max_ty = int(max_ty)
        self.sample_size = int(sample_size)
        self.epochs = int(epochs)
        self.goal_accuracy = 0.9
        self.callbacks = [
            print_accuracy_cb,
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
        accuracy = 0
        w1 = self.model.get_weights()
        while(self.epochs != 0 and accuracy <self.goal_accuracy):
            self.epochs -= 1
            w0 = w1
            self.model.fit(xs,ys,batch_size=self.batch_size, callbacks=self.callbacks)
            if self.epochs%100 ==0:
                w1 = self.model.get_weights()
                for l0,l1 in zip(w0,w1):
                    diff = l1-l0
                    sadiff = np.sum(np.abs(diff))
                    saw1 = np.sum(np.abs(l1))
                    print('sum(abs(grads))','%1.5E'%sadiff,
                        'sum(grads)','%1.5E'%np.sum(diff),
                        'sum(abs(weights))', '%1.5E'%saw1, 
                        'zeros', len(np.where(l1 == 0)), 
                        l1.shape)
                if self.save_file:
                    self.model.save(self.save_file)

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
