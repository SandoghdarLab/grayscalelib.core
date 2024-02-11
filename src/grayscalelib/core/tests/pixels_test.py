
from grayscalelib.core.pixels import Pixels, pixels_type


def pixels_test(cls: type[Pixels]):
    with pixels_type(cls):
        instance_creation()

def instance_creation():
    px = Pixels( list(range(256)))
    pass

