from norm_gtts import norm_gtts

def test_norm_gTTS():
    samples = norm_gtts('testando')
    assert len(samples) > 20000
    assert max(samples) <= 1
    assert min(samples) >= -1
