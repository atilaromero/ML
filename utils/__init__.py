import numpy as np

def one_hot(arr, num_categories):
    arr_shape = np.shape(arr)
    flatten = np.reshape(arr, -1)
    r = np.zeros((len(flatten),num_categories))
    r[np.arange(len(flatten)),flatten] = 1
    return r.reshape((*arr_shape,num_categories))

