from grayscalelib.core.simplepixels import SimplePixels

def test_simplepixels_init():
    for fbits in range(48):
        value = 2 ** (-fbits)
        assert SimplePixels(value, fbits=fbits).to_array()[()] == value
    # Ensure that each 

