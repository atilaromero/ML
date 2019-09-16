import os
import datetime
import random
from collections import OrderedDict

from tensorflow.keras.callbacks import EarlyStopping

import models
from block_sampler import BlockSampler
from dataset import Dataset
from batch_encoder import BatchEncoder
import callbacks
import report


def mk_result_dir():
    time_dir = datetime.datetime.now().isoformat()[:19].replace(':', '-')
    model_dir = os.path.join('results', time_dir)
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
    rdataset = rdataset.filter(lambda f: rdataset.category_from(f) in pdataset.categories)
    n = len(pdataset.categories)
    result_dir = mk_result_dir()

    for dataset, name in [(pdataset, 'purified'), (rdataset, 'raw')]:
        tset, vset = dataset.rnd_split_fraction(0.5)

        tsampler = BlockSampler(tset, group_by)
        tbenc = BatchEncoder(tsampler, 100, xs_encoder=xs_encoder)

        vsampler = BlockSampler(vset, group_by)
        vbenc = BatchEncoder(vsampler, 100, xs_encoder=xs_encoder)

        model = models.C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D(
            n, 8, 'softmax', 'categorical_crossentropy')
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
                                                        min_delta=1e-02, patience=2),
                                          # TensorBoard(
                                          #     log_dir=os.path.join(log_dir, model.name),
                                          #     # update_freq=3100,
                                          # ),
                                      ],
                                      use_multiprocessing=False,
                                      workers=0,
                                      )
        xs, ys = vbenc.__next__()
        ys_pred = model.predict(xs)

        report_data = OrderedDict()
        report_data['dataset'] = name
        report_data['classes'] = n
        reports = [
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
        for f in reports:
            d = f(**kwargs)
            report_data.update(d)
        report.save_report(report_data, result_dir + "/experiments.tsv")
        # xs, ys = dev_blocks.__next__()
        # confz = confusion(model, xs, ys, dev_blocks.ix_to_cat)
        # save_experiment_results(os.path.join(model_dir, 'confusion.txt'), confz)


if __name__ == '__main__':
    main()
