from fft import spectrogram_from_file

import matplotlib.pyplot as plt
def test_original():
    samples = spectrogram_from_file('dataset/pim.wav')
    assert samples.shape == (65,221)
    plt.imshow(samples.T)
    plt.gca().invert_yaxis()
    plt.savefig('samples.png')