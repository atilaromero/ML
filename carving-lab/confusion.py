import numpy as np


def confusion(model, xs, ys, ix_to_cat, thresholds=None, percent=False):
    results = []
    ys_pred = model.predict(xs)
    if thresholds is not None:
        ys_pred[ys_pred < thresholds] = 0
    for ix in ix_to_cat:
        result = {"True": ix_to_cat[ix]}
        ys_predix = ys_pred[np.argmax(ys, axis=-1) == ix]
        total = np.sum(np.argmax(ys, axis=-1) == ix)
        for jx in ix_to_cat:
            ypjs = ys_predix[np.argmax(ys_predix, axis=-1) == jx]
            count = len(ypjs)
            result[ix_to_cat[jx]] = count
            if percent:
                result[ix_to_cat[jx]] /= total
        results.append(result)
    return results
