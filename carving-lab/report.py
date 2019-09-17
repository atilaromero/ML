
def report_elapsed(elapsed, **kwargs):
    m, s = divmod(elapsed, 60)
    return {
        'Time': "{:d}m{:02d}s".format(int(m), int(s)),
    }


def report_epochs(history, **kwargs):
    epochs_count = len(history.epoch)
    return {
        'Epochs': epochs_count,
    }


def report_metrics(history, metrics, **kwargs):
    result = {}
    for metric in metrics:
        result[metric] = history.history[metric][-1]
    return result


def report_name(model, **kwargs):
    return {
        'Name': model.name,
    }


def save_report(data, tsv_path):
    save_reports([data], tsv_path)


def save_reports(many_data, tsv_path):
    keys = many_data[0].keys()
    with open(tsv_path, 'a') as f:
        f.write('\t'.join(keys))
        f.write('\n')
        for r in many_data:
            values = [str(r[k]) for k in keys]
            f.write('\t'.join(values))
            f.write('\n')
