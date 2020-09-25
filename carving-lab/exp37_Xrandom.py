import matplotlib.pyplot as plt
import os
import datetime
import random
from collections import OrderedDict
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.contrib.metrics import streaming_curve_points
from sklearn.metrics import precision_recall_curve, auc

import models
from block_sampler import BlockSampler
from dataset import Dataset
from batch_encoder import BatchEncoder, xs_encoder_8bits_11
import callbacks
import report

import matplotlib
matplotlib.use('Agg')

EXPERIMENT = 37

"""
same as 36, but with different dataset and sigmoid
"""


def mk_result_dir():
    time_dir = datetime.datetime.now().isoformat()[:19].replace(':', '-')
    model_dir = os.path.join('results', 'exp%s' % EXPERIMENT, time_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def main():
    xrandom(
        raw_dataset='../datasets/200jpgs',
        random_dataset='../datasets/random',
        filteredG_dataset='../datasets/filtering/good',
        filteredB_dataset='../datasets/filtering/bad',
        minimum=100,
        group_by='by_file',
        xs_encoder='8bits_11',
        validation_steps=10,
        steps_per_epoch=10,
        epochs=10000000,
        max_seconds=10*60,
    )


def xrandom(
    raw_dataset,
    random_dataset,
    filteredG_dataset,
    filteredB_dataset,
    minimum,
    group_by,
    xs_encoder,
    validation_steps,
    steps_per_epoch,
    epochs,
    max_seconds,
):
    raw_dset = Dataset.new_from_folders(raw_dataset)
    rnd_dset = Dataset.new_from_folders(random_dataset)
    rnd_dset = next(rnd_dset.rnd_split_num(200))
    raw_dset = raw_dset.filter_min_max(minimum)
    by_category = raw_dset.by_category()
    result_dir = mk_result_dir()

    reports = []
    for cat, filenames in by_category.items():
        dataset = Dataset(filenames, categories=[cat, 'zzz'])
        dataset = dataset.join(rnd_dset, categories=[cat, 'zzz'])
        tset, vset = dataset.rnd_split_fraction_by_category(0.5)

        tsampler = BlockSampler(tset, group_by)
        tbenc = BatchEncoder(tsampler, 100, xs_encoder=xs_encoder)

        vsampler = BlockSampler(vset, group_by)
        vbenc = BatchEncoder(vsampler, 100, xs_encoder=xs_encoder)

        model = models.C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D(
            2, 8, 'sigmoid', 'categorical_crossentropy')
        model.summary()

        timeIt = callbacks.TimeIt()

        history = model.fit_generator(iter(tbenc),
                                      validation_data=iter(vbenc),
                                      validation_steps=validation_steps,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      callbacks=[
            timeIt,
            # callbacks.SaveModel(os.path.join(result_dir, model.name + '.h5')),
            callbacks.TimeLimit(max_seconds),
            EarlyStopping(monitor='val_categorical_accuracy',
                          min_delta=1e-02, patience=4),
            # TensorBoard(
            #     log_dir=os.path.join(log_dir, model.name),
            #     # update_freq=3100,
            # ),
        ],
            use_multiprocessing=False,
            workers=0,
        )

        g_counter=0
        b_counter=0
        for filename in filenames:
            outdirG = os.path.join(filteredG_dataset, cat)
            outdirB = os.path.join(filteredB_dataset, cat)
            os.makedirs(outdirG, exist_ok=True)
            os.makedirs(outdirB, exist_ok=True)
            outpathG = os.path.join(outdirG, os.path.basename(filename))
            outpathB = os.path.join(outdirB, os.path.basename(filename))
            with open(filename, 'rb') as infile:
                with open(outpathG, 'wb') as outfileG:
                    with open(outpathB, 'wb') as outfileB:
                        while(True):
                            b = infile.read(51200)
                            if len(b) == 0:
                                break
                            bs = []
                            for i in range(0, len(b), 512):
                                b2 = b[i:i+512]
                                blk = np.zeros((512,), dtype='int')
                                blk[:len(b2)] = [int(x) for x in b2]
                                bs.append(blk)
                            ys = model.predict(xs_encoder_8bits_11(bs))
                            for i, y in enumerate(ys):
                                if np.argmax(y, axis=-1) == 1:
                                    outfileB.write(b[i:i+512])
                                    b_counter+=1
                                else:
                                    outfileG.write(b[i:i+512])
                                    g_counter+=1

        print()
        print(cat, history.history['val_categorical_accuracy'][-1])
        print()

        report_data = OrderedDict()
        report_data['category'] = cat
        report_data['good'] = g_counter
        report_data['bad'] = b_counter
        report_data['g/total'] = g_counter/(g_counter+b_counter)
        report_funcs = [
            # report.report_name,
            report.report_epochs,
            report.report_elapsed,
            report.report_metrics,
        ]
        kwargs = {
            'model': model,
            'history': history,
            'metrics': ['val_binary_accuracy', 'val_categorical_accuracy'],
            'elapsed': timeIt.elapsed,
        }
        for f in report_funcs:
            d = f(**kwargs)
            report_data.update(d)
        reports.append(report_data)
    report.save_reports(reports, result_dir + "/experiments.tsv")


if __name__ == '__main__':
    main()