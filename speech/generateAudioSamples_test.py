from generateAudioSamples import generateAudioSamples

def test_gTTS():
    samples = generateAudioSamples('testando')
    assert len(samples) > 20000
    assert max(samples) <= 1
    assert min(samples) >= -1
