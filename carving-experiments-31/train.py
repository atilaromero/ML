import os
import sys
import datetime
import random

from tensorflow.keras.callbacks import EarlyStopping

import models
from block_sampler import BlockSampler
from dataset import Dataset, mk_minimum_filter
from batch_encoder import BatchEncoder
import callbacks


def mk_result_dir():
    time_dir = datetime.datetime.now().isoformat()[:19].replace(':', '-')
    model_dir = os.path.join('results', time_dir)
    return model_dir


def main():
    nclasses(
        dataset_folder='../datasets/purified2',
        minimum=100,
        group_by='by_file',
        xs_encoder='8bits_11',
        validation_steps=10,
        steps_per_epoch=10,
        epochs=10000000,
        max_seconds=10*60,
    )


def nclasses(
    dataset_folder,
    minimum,
    group_by,
    xs_encoder,
    validation_steps,
    steps_per_epoch,
    epochs,
    max_seconds,
):
    dataset = Dataset.new_from_folders()
    dataset = dataset.filter_min_max(minimum)
    by_category = dataset.by_category()
    maximum = min([len(by_category[x]) for x in by_category])
    dataset = dataset.filter_min_max(minimum, maximum)

    for n in range(2, len(dataset.categories)+1):
        by_category = dataset.by_category()
        filenames = set()
        for k in random.sample(by_category.keys(), n):
            filenames = filenames.join(by_category[k])
        dsetN = Dataset(filenames)
        tset, vset = dsetN.rnd_split_fraction(0.5)

        tsampler = BlockSampler(tset, group_by)
        tbenc = BatchEncoder(tsampler, 100, xs_encoder=xs_encoder)

        vsampler = BlockSampler(vset, group_by)
        vbenc = BatchEncoder(vsampler, 100, xs_encoder=xs_encoder)

        model = models.C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D(
            n, 8, 'softmax', 'categorical_crossentropy')
        result_dir = mk_result_dir()
        model.summary()

        timeIt = callbacks.TimeIt()

        history = model.fit_generator(tbenc,
                                      validation_data=vbenc,
                                      validation_steps=validation_steps,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      callbacks=[
                                          timeIt,
                                          # callbacks.SaveModel(os.path.join(model_dir, model.name + '.h5')),
                                          callbacks.TimeLimit(max_seconds),
                                          EarlyStopping(monitor='val_categorical_accuracy',
                                                        min_delta=1e-02, patience=2),
                                          # TensorBoard(
                                          #     log_dir=os.path.join(log_dir, model.name),
                                          #     # update_freq=3100,
                                          # ),
                                      ],
                                      use_multiprocessing=False,
                                      workers=0,
                                      )

        # print(result)
        # results.append(result)
        # xs, ys = dev_blocks.__next__()
        # confz = confusion(model, xs, ys, dev_blocks.ix_to_cat)
        # print(confz)
        # save_experiment_results(os.path.join(model_dir, 'confusion.txt'), confz)
        # save_experiment_results(os.path.join(model_dir, 'experiments.tsv'), results)


if __name__ == '__main__':
    main()
