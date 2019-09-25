import os
import datetime

from dataset import Dataset
import report
from report import Reporter
import models

from trainer import Trainer

EXPERIMENT = 36

"""
saves discarted sectors too
"""



def main():
    xrandom(
        raw_dataset='../datasets/200jpgs',
        random_dataset='../datasets/random',
        filteredG_dataset='/mnt/teste/filtering/good',
        filteredB_dataset='/mnt/teste/filtering/bad',
        minimum=200,
        maximum=200,
    )


def xrandom(
    raw_dataset,
    random_dataset,
    filteredG_dataset,
    filteredB_dataset,
    minimum,
    maximum,
):
    raw_dset = Dataset.new_from_folders(raw_dataset)
    rnd_dset = Dataset.new_from_folders(random_dataset)
    rnd_dset = rnd_dset.filter_min_max(minimum, maximum)
    raw_dset = raw_dset.filter_min_max(minimum, maximum)
    by_category = raw_dset.by_category()
    result_dir = mk_result_dir()

    r = Reporter()
    for cat, filenames in by_category.items():
        print(cat, len(filenames))
        dataset = Dataset(filenames, categories=[cat, 'zzz'])
        dataset = dataset.join(rnd_dset, categories=[cat, 'zzz'])
        tset, vset = dataset.rnd_split_fraction_by_category(0.5)

        model = models.C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D(
            2, 8, 'softmax', 'categorical_crossentropy')
        model.summary()
        result = Trainer(model).train(tset, vset)

        print(cat, result.history.history['val_categorical_accuracy'][-1])

        filtered = filter_dataset(result.model, filenames)

        g_counter = 0
        b_counter = 0
        for v in filtered.values():
            g_counter += len(v[v == 0])
            b_counter += len(v[v == 1])

        save_filtered(os.path.join(result_dir, 'filtered.txt'))

        r.add(result,
              category=cat,
              good=g_counter,
              bad=b_counter,
              g_perc=g_counter/(g_counter+b_counter),
              **report.report_epochs(**result._asdict()),
              **report.report_elapsed(**result._asdict()),
              **report.report_metrics(**result._asdict()),
              )
    r.save_report(result_dir + "/experiments.tsv")





if __name__ == '__main__':
    main()
