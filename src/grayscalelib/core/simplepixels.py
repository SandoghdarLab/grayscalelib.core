from __future__ import annotations

from typing import Callable, Literal, Self

import itertools

import math

import operator

from grayscalelib.core.protocols import ArrayLike, RealLike

from grayscalelib.core.simplearray import SimpleArray

from grayscalelib.core.pixels import Pixels, set_default_pixels_type


class SimplePixels(Pixels):
    """A reference implementation of the Pixels protocol.

    Ideally, this class can be used as a starting point for more elaborate
    implementations, and as a reference solutions for testing.  With these
    use-cases in mind, the main goals of this code are correctness, simplicity,
    and having no additional dependencies.
    """
    _shape: tuple[int, ...]
    _ibits: int
    _fbits: int
    _array: SimpleArray[int]

    def _init_(
            self,
            array: ArrayLike,
            ibits: int,
            fbits: int,
            scale: RealLike,
            offset: RealLike,
            maxval: int):
        shape = array.shape
        values: list[int] = []
        for index in itertools.product(*tuple(range(n) for n in shape)):
            value = max(0, min(round(array[*index] * scale + offset), maxval))
            assert isinstance(value, int)
            values.append(value)
        self._shape = shape
        self._ibits = ibits
        self._fbits = fbits
        self._array = SimpleArray(values, shape)

    # The abstract Pixels class has an encoding priority of zero.  By returning
    # one here, we ensure that this reference implementation takes precedence.
    @classmethod
    def __encoding_priority__(cls):
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ibits(self) -> int:
        return self._ibits

    @property
    def fbits(self) -> int:
        return self._fbits

    @property
    def numerators(self) -> SimpleArray[int]:
        return self._array

    def to_array(self) -> SimpleArray[float]:
        d = self.denominator
        values = list(map(lambda n: n / d, self._array.values))
        return SimpleArray(values, self.shape)

    def _getitem_(self, index) -> Self:
        ibits = self._ibits
        fbits = self._fbits
        array = self._array
        selection = array[index]
        if selection is array:
            return self
        if isinstance(selection, int):
            selection = SimpleArray([selection], ())
        return self.from_numerators(selection, ibits, fbits)

    def _permute_(self, permutation) -> Self:
        ibits = self._ibits
        fbits = self._fbits
        array = self._array
        return type(self).from_numerators(array.permute(permutation), ibits, fbits)

    def _broadcast_to_(self, shape) -> Self:
        ibits = self._ibits
        fbits = self._fbits
        array = self._array
        old_shape = self._shape
        new_shape = shape
        growth = math.prod(new_shape[len(old_shape):])
        new_values: list[int] = []
        for value in array.values:
            new_values.extend([value] * growth)
        new_array = SimpleArray(new_values, new_shape)
        return type(self).from_numerators(new_array, ibits, fbits)

    def _bool_(self) -> bool:
        return any(v > 0 for v in self._array.values)

    def _map1(self, ibits, fbits, fn) -> Self:
        ibits = self._ibits
        fbits = self._fbits
        array = self._array
        return type(self).from_numerators(array.map1(fn), ibits, fbits)

    def _map2(self, other, ibits, fbits, fn) -> Self:
        ibits = self._ibits
        fbits = self._fbits
        array = self._array
        return type(self).from_numerators(array.map2(fn, other._array), ibits, fbits)

    def _not_(self):
        return self._map1(1, 0, pixel_not)

    def _and_(self, other: SimplePixels) -> Self:
        return self._map2(other, 1, 0, pixel_and)

    def _or_(self, other: SimplePixels) -> Self:
        return self._map2(other, 1, 0, pixel_or)

    def _xor_(self, other: SimplePixels) -> Self:
        return self._map2(other, 1, 0, pixel_xor)

    def _lshift_(self, amount: int) -> Self:
        ibits = max(0, self.ibits + amount)
        fbits = max(0, self.fbits - amount)
        return self._map1(ibits, fbits, lambda x: x << amount)

    def _rshift_(self, amount: int) -> Self:
        ibits = max(0, self.ibits + amount)
        fbits = max(0, self.fbits - amount)
        return self._map1(ibits, fbits, lambda x: x >> amount)

    def _invert_(self) -> Self:
        d = self.denominator
        return self._map1(self.ibits, self.fbits, lambda x: max(0, d - x))

    def _add_(self, other: Self) -> Self:
        xfbits = self.fbits
        yfbits = other.fbits
        fbits = max(self.fbits, other.fbits)
        maxval = (1 << fbits) - 1
        if xfbits == yfbits:
            fn = lambda x, y: min(x + y, maxval)
        elif xfbits < yfbits:
            shift = yfbits - xfbits
            fn = lambda x, y: min((x << shift) + y, maxval)
        else:
            shift = xfbits - yfbits
            fn = lambda x, y: min(x + (y << shift), maxval)
        return self._map2(other, 1, fbits, fn)

    def _sub_(self, other: Self) -> Self:
        xfbits = self.fbits
        yfbits = other.fbits
        fbits = max(self.fbits, other.fbits)
        if xfbits == yfbits:
            fn = lambda x, y: max(0, x - y)
        elif xfbits < yfbits:
            shift = yfbits - xfbits
            fn = lambda x, y: max(0, (x << shift) - y)
        else:
            shift = xfbits - yfbits
            fn = lambda x, y: max(0, x - (y << shift))
        return self._map2(other, 1, fbits, fn)

    def _mul_(self, other: Self) -> Self:
        xfbits = self.fbits
        yfbits = other.fbits
        if xfbits <= yfbits:
            fbits = yfbits
            shift = xfbits
        else:
            fbits = xfbits
            shift = yfbits
        # Compute z such that the later right shift rounds correctly.
        z = 0 if shift == 0 else (1 << (shift - 1))
        return self._map2(other, 1, fbits, lambda x, y: (x * y + z) >> shift)

    def _pow_(self, power: Self) -> Self:
        raise NotImplementedError()

    def _truediv_(self, other: Self) -> Self:
        raise NotImplementedError()

    def _floordiv_(self, other: Self) -> Self:
        raise NotImplementedError()

    def _mod_(self, other: Self) -> Self:
        raise NotImplementedError()

    def _cmp(self, other: Self, op: Callable[[int, int], Literal[0, 1]]) -> Self:
        xfbits = self.fbits
        yfbits = other.fbits
        if xfbits == yfbits:
            fn = lambda x, y: op(x, y)
        elif xfbits < yfbits:
            shift = yfbits - xfbits
            fn = lambda x, y: op(x << shift, y)
        else:
            shift = xfbits - yfbits
            fn = lambda x, y: op(x, y << shift)
        return self._map2(other, 1, 0, fn)

    def _lt_(self, other: Self) -> Self:
        return self._cmp(other, operator.lt)

    def _gt_(self, other: Self) -> Self:
        return self._cmp(other, operator.gt)

    def _le_(self, other: Self) -> Self:
        return self._cmp(other, operator.le)

    def _ge_(self, other: Self) -> Self:
        return self._cmp(other, operator.ge)

    def _eq_(self, other: Self) -> Self:
        return self._cmp(other, operator.eq)


def pixel_not(x) -> Literal[0, 1]:
    return 1 if x == 0 else 0

def pixel_and(x, y) -> Literal[0, 1]:
    return 1 if (x > 0 and y > 0) else 0

def pixel_or(x, y) -> Literal[0, 1]:
    return 1 if (x > 0 or y > 0) else 0

def pixel_xor(x, y) -> Literal[0, 1]:
    return 1 if (x > 0 and y == 0) or (x == 0 and y > 0) else 0

set_default_pixels_type(SimplePixels)
