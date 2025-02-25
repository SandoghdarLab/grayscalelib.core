from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Self, TypeVar

from itertools import product

from math import prod

import numpy as np

import numpy.typing as npt

from grayscalelib.core.pixels import Discretization, Initializer, ConcretePixels, ConcretePixelsInitializer, register_default_pixels_type, boolean_discretization


T = TypeVar('T')

uint = np.uint8 | np.uint16 | np.uint32 | np.uint64


class NumpyPixels(ConcretePixels):
    """A reference implementation of the Pixels protocol.

    Ideally, this class can be used as a starting point for more elaborate
    implementations, and to create reference solutions for testing.  With these
    use-cases in mind, the main goals of this code are correctness, simplicity,
    and having Numpy as the only dependency.
    """
    _raw: npt.NDArray[uint]

    @classmethod
    def _initializer_(
            cls: type[T],
            data: npt.ArrayLike,
            discretization: Discretization
    ) -> Initializer[T]:
        dtype = integer_dtype(*discretization.codomain)
        raw = np.vectorize(discretization, otypes=[dtype])(data)
        return NumpyPixelsInitializer(raw.shape, discretization, raw)

    @classmethod
    def __encoding_priority__(cls):
        return 0 # Any other implementation should take precedence.

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return np.vectorize(self.discretization.inverse, otypes=[np.float64])(self._raw)

    @property
    def raw(self) -> npt.NDArray[uint]:
        return self._raw

    def _getitem_(self, index) -> Self:
        discretization = self.discretization
        raw = self._raw.__getitem__(index)
        return type(self)(NumpyPixelsInitializer(raw.shape, discretization, raw))

    def _permute_(self, permutation) -> Self:
        raw = self._raw.transpose(permutation)
        return type(self)(NumpyPixelsInitializer(raw.shape, self.discretization, raw))

    def _align_with_(self, black: float, white: float, states: int) -> Self:
        i2f = self.discretization.inverse
        f2i = Discretization((black, white), (0, max(0, states-1)))
        a = i2f.a * f2i.a
        b = i2f.b * f2i.a + f2i.b
        i1, i2 = round(i2f.domain.lo * a + b), round(i2f.domain.hi * a + b)
        f1, f2 = f2i.inverse(i1), f2i.inverse(i2)
        discretization = Discretization((f1, f2), (i1, i2))
        dtype = integer_dtype(i1, i2)
        if a == 0.0 and b == 0.0:
            raw = np.zeros(self._raw.shape, dtype=dtype)
        elif a == 0.0:
            raw = np.broadcast_to(np.rint(b).astype(dtype), self.shape)
        elif a == 1.0:
            raw = np.rint(self._raw + b).astype(dtype)
        else:
            raw = np.rint(self._raw * a + b).astype(dtype)
        return type(self)(NumpyPixelsInitializer(raw.shape, discretization, raw))

    def _reshape_(self, shape) -> Self:
        raw = self._raw.reshape(shape)
        return type(self)(NumpyPixelsInitializer(raw.shape, self.discretization, raw))

    def _broadcast_to_(self, shape) -> Self:
        padding = (1,) * max(0, len(shape) - self.rank)
        padded = np.reshape(self._raw, self.shape + padding)
        raw = np.broadcast_to(padded, shape)
        return type(self)(NumpyPixelsInitializer(raw.shape, self.discretization, raw))

    def _any_(self) -> bool:
        false = self.discretization(self.black)
        return bool(np.any(self._raw != false))

    def _all_(self) -> bool:
        false = self.discretization(self.black)
        return bool(np.all(self._raw != false))

    def _and_(self, other: NumpyPixels) -> Self:
        a = self._raw != self.discretization(self.black)
        b = other._raw != other.discretization(other.black)
        raw = np.logical_and(a, b).astype(np.uint8)
        return type(self)(NumpyPixelsInitializer(raw.shape, boolean_discretization, raw))

    def _or_(self, other: NumpyPixels) -> Self:
        a = self._raw != self.discretization(self.black)
        b = other._raw != other.discretization(other.black)
        array = np.logical_or(a, b).astype(np.uint8)
        return type(self)(NumpyPixelsInitializer(array.shape, boolean_discretization, array))

    def _xor_(self, other: NumpyPixels) -> Self:
        a = self._raw != self.discretization(self.black)
        b = other._raw != other.discretization(other.black)
        array = np.logical_xor(a, b).astype(np.uint8)
        return type(self)(NumpyPixelsInitializer(array.shape, boolean_discretization, array))

    def _invert_(self) -> Self:
        raw =  self._discretization.codomain.hi - self._raw
        return type(self)(NumpyPixelsInitializer(raw.shape, self.discretization, raw))

    def _add_(self, other: Self) -> Self:
        d1 = self.discretization
        d2 = other.discretization
        fmin = d1.domain.lo + d2.domain.lo
        fmax = d1.domain.hi + d2.domain.hi
        imin = d1.codomain.lo + d2.codomain.lo
        imax = d1.codomain.hi + d2.codomain.hi
        discretization = Discretization((fmin, fmax), (imin, imax))
        dtype = integer_dtype(imin, imax)
        raw = np.add(self._raw, other.raw, dtype=dtype)
        return type(self)(NumpyPixelsInitializer(raw.shape, discretization, raw))

    def _sub_(self, other: Self) -> Self:
        pass # TODO

    def _mul_(self, other: Self) -> Self:
        pass # TODO

    def _pow_(self, other: float) -> Self:
        pass # TODO

    def _truediv_(self, other: Self) -> Self:
        pass # TODO

    def _mod_(self, other: Self) -> Self:
        pass # TODO

    def _lt_(self, other: Self) -> Self:
        pass # TODO

    def _gt_(self, other: Self) -> Self:
        pass # TODO

    def _le_(self, other: Self) -> Self:
        pass # TODO

    def _ge_(self, other: Self) -> Self:
        pass # TODO

    def _eq_(self, other: Self) -> Self:
        pass # TODO

    def _ne_(self, other: Self) -> Self:
        pass # TODO

    def _rolling_sum_(self, window_sizes):
        array = self._raw
        for axis, window_size in enumerate(window_sizes):
            shape = array.shape
            oldsize = shape[axis]
            newsize = oldsize - window_size + 1
            values: list[int] = []
            for prefix in product(*[range(s) for s in shape[:axis]]):
                rest = shape[axis+1:]
                suffixes = list(product(*[range(s) for s in rest])) or (())
                sums = [0] * prod(rest)
                # Compute the sums of the first window_size elements.
                for index in range(window_size):
                    for pos, suffix in enumerate(suffixes):
                        sums[pos] += array.item((*prefix, index, *suffix))
                # Initialize the first entry of the result.
                values.extend(sums)
                # Compute the remaining entries of the result by sliding the
                # window one element at a time.
                for index in range(1, newsize):
                    for pos, suffix in enumerate(suffixes):
                        sums[pos] -= array.item((*prefix, index-1, *suffix))
                        sums[pos] += array.item((*prefix, index+window_size-1, *suffix))
                    values.extend(sums)
            array = NumpyArray(values, shape[:axis] + (newsize,) + shape[axis+1:])
        limit = prod(window_sizes) * self.limit
        return self.from_raw(array, self.power, limit)


register_default_pixels_type(NumpyPixels)


NP = TypeVar('NP', bound=NumpyPixels)


@dataclass(frozen=True)
class NumpyPixelsInitializer(ConcretePixelsInitializer[NP]):
    raw: npt.NDArray[uint]

    def initialize(self, /, instance: NP):
        super().initialize(instance)
        instance._raw = self.raw


def integer_dtype(*integers: int) -> npt.DTypeLike:
    """The smallest integer dtype that encompasses all the supplied integers."""
    if len(integers) == 0:
        return np.uint8
    lo = min(integers)
    hi = max(integers)
    if lo < 0:
        dtypes = [np.int8, np.int16, np.int32, np.int64]
    else:
        dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in dtypes:
        ii = np.iinfo(dtype)
        if ii.min <= lo and hi <= ii.max:
            return dtype
    raise TypeError(f"No dtype large enough for the following integers: {integers}.")
