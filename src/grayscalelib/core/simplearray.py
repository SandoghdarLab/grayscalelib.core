from __future__ import annotations
from itertools import product

from math import prod

from typing import Callable, Generic, TypeVar

from grayscalelib.core.protocols import ArrayLike

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")

class SimpleArray(ArrayLike, Generic[T]):
    """
    A simple, n-dimensional, immutable array class.
    """
    _shape: tuple[int, ...]
    _values: list[T]
    _strides: tuple[int, ...]

    def __init__(self, values: list[T], shape: tuple[int, ...] | None = None):
        if shape is None:
            shape = (len(values),)
        assert isinstance(values, list)
        assert isinstance(shape, tuple)
        for size in shape:
            assert isinstance(size, int)
        assert len(values) == prod(shape)
        rank = len(shape)
        self._shape = shape
        self._values = values
        self._strides = tuple(prod(shape[n:]) for n in range(1, rank+1))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def values(self) -> list[T]:
        return self._values

    def __repr__(self) -> str:
        """
        A textual representation of this array.
        """
        def listify(prefix, sizes):
            if len(sizes) == 0:
                return self[*prefix]
            elif len(sizes) == 1:
                return [self[*prefix, i] for i in range(sizes[0])]
            else:
                return [listify(prefix + (i,), sizes[1:]) for i in range(sizes[0])]

        return f"SimpleArray({listify((), self.shape)})"

    def __getitem__(self, subscripts) -> T | SimpleArray[T]:
        if subscripts == ...:
            subscripts = ()
        if not isinstance(subscripts, tuple):
            subscripts = (subscripts,)
        old_shape = self.shape
        old_values: list[T] = self._values
        old_strides = self.strides
        old_rank = len(old_shape)
        nsubscripts = len(subscripts)
        assert nsubscripts <= old_rank
        # Compute the iteration space.
        ranges: list[range] = []
        sizes: list[int] = []
        for index, size, stride in zip(subscripts, old_shape, old_strides):
            if isinstance(index, int):
                if not (0 <= index < size):
                    raise IndexError(f"Out-of-bounds index {index} for axis of size {size}.")
                pos = index * stride
                ranges.append(range(pos, pos+1))
            if isinstance(index, slice):
                start, stop, step = index.indices(size)
                rstart = start * stride
                rstop = stop * stride
                rstep = 1 if stride == 0 else step * stride
                ranges.append(range(rstart, rstop, rstep))
                sizes.append(len(range(start, stop, step)))
        for size, stride in zip(old_shape[nsubscripts:], old_strides[nsubscripts:]):
            ranges.append(range(0, size * stride, 1 if stride == 0 else stride))
            sizes.append(size)
        # Derive the new shape and new values from the iteration space.
        new_shape = tuple(sizes)
        new_values: list[T] = []
        for parts in product(*ranges):
            new_values.append(old_values[sum(parts)])
        if new_shape == ():
            return new_values[0]
        else:
            return SimpleArray(new_values, new_shape)

    def permute(self, axes: tuple[int, ...]):
        old_shape = self.shape
        old_strides = self.strides
        old_values = self._values
        naxes = len(axes)
        new_shape = tuple(old_shape[ax] for ax in axes) + old_shape[naxes:]
        if len(old_values) == 0:
            return SimpleArray([], new_shape)
        tmp_strides = tuple(old_strides[ax] for ax in axes) + old_strides[naxes:]
        ranges: list[range] = []
        for size, stride in zip(new_shape, tmp_strides):
            ranges.append(range(0, size * stride, stride))
        new_values = []
        for parts in product(*ranges):
            new_values.append(old_values[sum(parts)])
        return SimpleArray(new_values, new_shape)

    def map1(self, fn: Callable[[T], R]) -> SimpleArray[R]:
        return SimpleArray(list(map(fn, self._values)), self.shape)

    def map2(self, fn: Callable[[T, U], R], other: SimpleArray[U]) -> SimpleArray[R]:
        return SimpleArray(list(map(fn, self._values, other._values)), self.shape)
