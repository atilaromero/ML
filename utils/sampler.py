import numpy as np

def shuffle(examples):
    result = examples[:]
    np.random.shuffle(result)
    return result

def choice(examples, sample_size):
    return np.random.choice(examples, size=sample_size, replace=True)