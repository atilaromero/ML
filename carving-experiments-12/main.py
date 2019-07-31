import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Activation, TimeDistributed, Flatten, Dot, Softmax, Lambda
import tensorflow.keras.backend as K

from run_experiments import Experiment, run_experiments, save_experiment_results

NINPUT=511
NOUTPUT=511

def L():
    last = l0 = Input(shape=(NINPUT,256))
    last = LSTM(256, return_sequences=True)(last)
    model = tf.keras.Model([l0], last)
    myfuncname = sys._getframe().f_code.co_name
    return Experiment(myfuncname, model)

experiments = [
    L()
]

if __name__ == '__main__':
    results = []
    for t in ['pdf', 'jpg', 'png']:
        e_temp = []
        for e in experiments:
            temp = Experiment(e.name+'-'+t, e.model)
            e_temp.append(temp)
        for r in run_experiments(e_temp,
            batch_size=10,
            validation_batch_size=10,
            validation_steps=100,
            steps_per_epoch=100,
            epochs=100000,
            trainDir='../datasets/carving/train/'+t,
            validationDir='../datasets/carving/dev/'+t,
            seconds_limit=6*60*60,
            acc_limit=None,
            val_acc_limit=1.0,
            ):
            print(r)
            results.append(r)

    save_experiment_results('experiments.tsv', results)
