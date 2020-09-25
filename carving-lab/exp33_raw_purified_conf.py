import os
import datetime
import random
from collections import OrderedDict
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.contrib.metrics import streaming_curve_points
from sklearn.metrics import roc_curve, auc

import models
from block_sampler import BlockSampler
from dataset import Dataset
from batch_encoder import BatchEncoder
import callbacks
import report
import confusion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Compares confusion matrix of models trainded with purified and unmodified datasets.

"""

EXPERIMENT = 33


def mk_result_dir():
    time_dir = datetime.datetime.now().isoformat()[:19].replace(':', '-')
    model_dir = os.path.join('results', 'exp%d' % EXPERIMENT, time_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def main():
    raw_purified(
        raw_dataset='../datasets/govdocs1/sample200',
        purified_dataset='../datasets/purified2',
        minimum=100,
        group_by='by_file',
        xs_encoder='8bits_11',
        validation_steps=10,
        steps_per_epoch=10,
        epochs=10000000,
        max_seconds=10*60,
    )


def raw_purified(
    raw_dataset,
    purified_dataset,
    minimum,
    group_by,
    xs_encoder,
    validation_steps,
    steps_per_epoch,
    epochs,
    max_seconds,
):
    pdataset = Dataset.new_from_folders(purified_dataset)
    rdataset = Dataset.new_from_folders(raw_dataset)
    pdataset = pdataset.filter_min_max(minimum)
    by_category = pdataset.by_category()
    maximum = min([len(by_category[x]) for x in by_category])
    pdataset = pdataset.filter_min_max(minimum, maximum)
    rdataset = rdataset.filter(
        lambda f: rdataset.category_from(f) in pdataset.categories)
    rdataset.rebuild_categories(pdataset.categories)
    n = len(pdataset.categories)
    result_dir = mk_result_dir()

    mymodels = {}
    vsets = {}
    for dataset, name in [(pdataset, 'purified'), (rdataset, 'raw')]:
        tset, vset = dataset.rnd_split_fraction(0.5)
        vsets[name] = vset

        tsampler = BlockSampler(tset, group_by)
        tbenc = BatchEncoder(tsampler, 100, xs_encoder=xs_encoder)

        vsampler = BlockSampler(vset, group_by)
        vbenc = BatchEncoder(vsampler, 100, xs_encoder=xs_encoder)

        model = models.C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D(
            n, 8, 'softmax', 'categorical_crossentropy')
        model.summary()
        mymodels[name] = model

        timeIt = callbacks.TimeIt()

        model.fit_generator(iter(tbenc),
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

    dsetname = 'raw'
    vsampler = BlockSampler(vsets[dsetname], group_by)
    vbenc = BatchEncoder(vsampler, 1000, xs_encoder=xs_encoder)
    xs, ys = vbenc.__next__()
    for modelname, model in mymodels.items():
        thresholds = np.zeros((len(vset.categories),))
        for ix in range(len(vset.categories)):
            ys_pred = model.predict(xs)
            fpr, tpr, thr = roc_curve(ys[:, ix], ys_pred[:, ix])
            score = tpr - fpr
            m = np.argmax(score, axis=-1)
            thresholds[ix] = thr[m]
        conf1 = confusion.confusion(model, xs, ys, vset.ix_to_cat, thresholds=thresholds, percent=False)
        conf2 = confusion.confusion(model, xs, ys, vset.ix_to_cat, thresholds=thresholds, percent=True)
        report.save_reports(conf1, result_dir + "/confusion-%s.tsv"%modelname)
        report.save_reports(conf2, result_dir + "/confusion-perc-%s.tsv"%modelname)


if __name__ == '__main__':
    main()