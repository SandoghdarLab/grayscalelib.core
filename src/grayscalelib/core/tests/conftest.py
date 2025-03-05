from grayscalelib.core import NumpyPixels, Pixels


def pytest_generate_tests(metafunc):
    if "pixels_subclass" in metafunc.fixturenames:
        metafunc.parametrize("pixels_subclass", [Pixels, NumpyPixels])
