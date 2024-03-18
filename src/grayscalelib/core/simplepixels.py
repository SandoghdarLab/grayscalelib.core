from __future__ import annotations

from typing import Callable, Literal, Self

from itertools import product

from math import prod

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
    _power: int
    _limit: int
    _array: SimpleArray[int]

    def _init_(
            self,
            array: ArrayLike,
            black: RealLike,
            white: RealLike,
            limit: RealLike,
            power: int):
        shape = array.shape
        scale = 1 / ((white - black) * (2**power))
        vmax = round((limit - black) * scale)
        values: list[int] = []
        for index in product(*tuple(range(n) for n in shape)):
            clipped = max(0, min(array[index] - black, limit - black))
            value = round(clipped * scale)
            assert 0 <= value <= vmax
            values.append(value)
        self._shape = shape
        self._power = power
        self._limit = vmax
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
    def power(self) -> int:
        return self._power

    @property
    def data(self):
        return self._array

    @property
    def limit(self) -> int:
        return self._limit

    def to_array(self) -> SimpleArray[float]:
        factor = 2**self.power
        values = list(map(lambda n: n * factor, self._array.values))
        return SimpleArray(values, self.shape)

    def _getitem_(self, index) -> Self:
        power = self._power
        limit = self._limit
        array = self._array
        selection = array[index]
        if selection is array:
            return self
        if isinstance(selection, int):
            selection = SimpleArray([selection], ())
        return self.from_data(selection, power, limit)

    def _permute_(self, permutation) -> Self:
        power = self._power
        limit = self._limit
        array = self._array
        return type(self).from_data(array.permute(permutation), power, limit)

    def _reshape_(self, shape) -> Self:
        power = self._power
        limit = self._limit
        array = SimpleArray(self._array.values, shape)
        return type(self).from_data(array, power, limit)

    def _broadcast_to_(self, shape) -> Self:
        limit = self._limit
        power = self._power
        array = self._array
        old_shape = self._shape
        new_shape = shape
        growth = prod(new_shape[len(old_shape):])
        new_values: list[int] = []
        for value in array.values:
            new_values.extend([value] * growth)
        new_array = SimpleArray(new_values, new_shape)
        return type(self).from_data(new_array, power, limit)

    def _any_(self) -> bool:
        return any(v > 0 for v in self._array.values)

    def _all_(self) -> bool:
        return all(v > 0 for v in self._array.values)

    def _map1(self, power, limit, fn) -> Self:
        array = self._array
        return type(self).from_data(array.map1(fn), power, limit)

    def _map2(self, other, power, limit, fn) -> Self:
        array = self._array
        return type(self).from_data(array.map2(fn, other._array), power, limit)

    def _and_(self, other: SimplePixels) -> Self:
        return self._map2(other, 0, 1, pixel_and)

    def _or_(self, other: SimplePixels) -> Self:
        return self._map2(other, 0, 1, pixel_or)

    def _xor_(self, other: SimplePixels) -> Self:
        return self._map2(other, 0, 1, pixel_xor)

    def _lshift_(self, amount: int) -> Self:
        return type(self).from_data(self.data, self.power + amount, self.limit)

    def _rshift_(self, amount: int) -> Self:
        return type(self).from_data(self.data, self.power - amount, self.limit)

    def _invert_(self) -> Self:
        if self.power <= 0:
            power = self.power
            white = self.white
            return self._map1(power, white, lambda x: max(0, white - x))
        else: # self.power > 0
            power = 0
            white = 1
            return self._map1(power, white, lambda x: 1 if x == 0 else 0)

    def _add_(self, other: Self) -> Self:
        xpower, ypower = self.power, other.power
        xlimit, ylimit = self.limit, other.limit
        power = min(xpower, ypower, 0)
        limit = min(round(xlimit * 2**(power - xpower) +
                          ylimit * 2**(power - ypower)),
                    2**(-power))
        xshift = xpower - power
        yshift = ypower - power
        fn = lambda x, y: min((x << xshift) + (y << yshift), limit)
        return self._map2(other, power, limit, fn)

    def _sub_(self, other: Self) -> Self:
        xpower, ypower = self.power, other.power
        xlimit = self.limit
        power = min(xpower, ypower, 0)
        limit = min(round(xlimit * 2**(xpower - power)), 2**(-power))
        xshift = xpower - power
        yshift = ypower - power
        fn = lambda x, y: max(0, (x << xshift) - (y << yshift))
        return self._map2(other, power, limit, fn)

    def _mul_(self, other: Self) -> Self:
        xpower, ypower = self.power, other.power
        power = min(xpower, ypower, 0)
        limit = 2**(-power)
        # x*2^xpower * y*2^ypower = x*y*2^(xpower + ypower)
        #                         = x*y*2^(xpower + ypower - power) * 2^power
        factor = 2**((xpower + ypower) - power)
        if isinstance(factor, int):
            fn = lambda x, y: min(x * y * factor, limit)
        else:
            fn = lambda x, y: min(round(x * y * factor), limit)
        return self._map2(other, power, limit, fn)

    def _pow_(self, other: float) -> Self:
        xpower, xlimit, y = self.power, self.limit, other
        power = min(xpower, 0)
        # v * 2^power = (x * 2^xpower)^y
        #           v = x^y * 2^(y * xpower - power)
        factor = 2**(y * xpower - power)
        limit = min(2**abs(power), round(xlimit**y * factor))
        fn = lambda x: min(round(x**y * factor), limit)
        return self._map1(power, limit, fn)

    def _truediv_(self, other: Self) -> Self:
        raise NotImplementedError()

    def _floordiv_(self, other: Self) -> Self:
        raise NotImplementedError()

    def _mod_(self, other: Self) -> Self:
        raise NotImplementedError()

    def _cmp(self, other: Self, op: Callable[[int, int], Literal[0, 1]]) -> Self:
        xpower, ypower = self.power, other.power
        if xpower == ypower:
            fn = lambda x, y: op(x, y)
        elif xpower < ypower:
            shift = ypower - xpower
            fn = lambda x, y: op(x, y << shift)
        else: # ypower < xpower
            shift = xpower - ypower
            fn = lambda x, y: op(x << shift, y)
        return self._map2(other, 0, 1, fn)

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

    def _ne_(self, other: Self) -> Self:
        return self._cmp(other, operator.ne)

    def _rolling_sum_(self, window_sizes):
        array = self._array
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
            array = SimpleArray(values, shape[:axis] + (newsize,) + shape[axis+1:])
        limit = prod(window_sizes) * self.limit
        return self.from_data(array, self.power, limit)


def pixel_not(x) -> Literal[0, 1]:
    return 1 if x == 0 else 0


def pixel_and(x, y) -> Literal[0, 1]:
    return 1 if (x > 0 and y > 0) else 0


def pixel_or(x, y) -> Literal[0, 1]:
    return 1 if (x > 0 or y > 0) else 0


def pixel_xor(x, y) -> Literal[0, 1]:
    return 1 if (x > 0 and y == 0) or (x == 0 and y > 0) else 0


def strip_bits(i: int, n: int):
    if n == 0:
        return i
    elif n > 0:
        return (i + (1 << (n - 1))) >> n
    else:
        raise ValueError("Negative number of stripped digits.")


set_default_pixels_type(SimplePixels)
