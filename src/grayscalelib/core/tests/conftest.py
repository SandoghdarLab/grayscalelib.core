from grayscalelib.core.pixels import Pixels
from grayscalelib.core.simplepixels import SimplePixels

def pytest_generate_tests(metafunc):
    if "pixels_subclass" in metafunc.fixturenames:
        metafunc.parametrize("pixels_subclass", [Pixels, SimplePixels])

