# grayscalelib.core

This project defines a protocol for working with grayscale images, videos and
containers of arbitrary rank.  It also provides a reference implementation in
NumPy.

## Installation

All releases are published on [PyPi](https://pypi.org), so all the popular
installation methods work out-of-the-box.  For example, here is how you can
install the library using pip.

```sh
python -m pip install grayscalelib.core
```

## Usage

The main data structure of this library is the `Pixels` class.  All other
functionality is accessible as methods and class methods of this class.

```python
from grayscalelib.core import Pixels

px = Pixels([0.00, 0.25, 0.50, 0.75 1.00])

# List all methods defined on Pixels
dir(px)
```
