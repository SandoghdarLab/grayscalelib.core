from __future__ import annotations

from abc import abstractmethod

from contextlib import contextmanager

from dataclasses import dataclass

from math import ceil, prod

from os import PathLike

from pathlib import Path

from types import EllipsisType

from typing import Generic, Protocol, Self, TypeVar, runtime_checkable

import numpy as np

import numpy.typing as npt

from grayscalelib.core.discretization import Discretization

from grayscalelib.core.encodable import choose_encoding, Encodable


###############################################################################
###
### Global Variables


_enforced_pixels_type: type[Pixels] | None = None

_default_pixels_type: type[Pixels] | None = None

_default_states: int = 256

boolean_discretization = Discretization((0.0, 1.0), (0, 1))

uint = np.uint8 | np.uint16 | np.uint32 | np.uint64


def register_default_pixels_type(cls: type[Pixels]):
    """Consider the supplied pixels class as a default implementation."""
    global _default_pixels_type
    if _default_pixels_type is None:
        _default_pixels_type = cls
    elif _default_pixels_type.__encoding_priority__() < cls.__encoding_priority__():
        _default_pixels_type = cls
    else:
        pass


def encoding(*clss: type[Pixels]) -> type[Pixels]:
    """Determine the canonical encoding of the supplied pixels classes."""
    if _enforced_pixels_type:
        return _enforced_pixels_type
    if _default_pixels_type is not None:
        clss += (_default_pixels_type,)
    if len(clss) == 0:
        raise RuntimeError("Not a single registered default Pixels type.")
    return choose_encoding(*clss)


@contextmanager
def pixels_type(pt: type[Pixels], /):
    """
    Create a context in which all operations on pixels will be carried out
    using the supplied representation.
    """
    global _enforced_pixels_type
    previous = _enforced_pixels_type
    _enforced_pixels_type = pt
    try:
        yield pt
    finally:
        _enforced_pixels_type = previous


@contextmanager
def default_pixels_type(pt: type[Pixels], /):
    """
    Create a context in which the supplied pixels representation takes
    precedence over representations with lower priority.
    """
    global _default_pixels_type
    previous = _default_pixels_type
    _default_pixels_type = pt
    try:
        yield pt
    finally:
        _default_pixels_type = previous


@contextmanager
def default_pixels_states(states: int):
    """
    Create a context in which the supplied number of states is the default
    when constructing pixels.

    """
    global _default_states
    previous = _default_states
    _default_states = states
    try:
        yield states
    finally:
        _default_states = previous


###############################################################################
###
### The Pixels Class


@runtime_checkable
class Real(Protocol):
    def __float__(self) -> float:
        ...


T = TypeVar('T')


@dataclass(frozen=True)
class Initializer(Generic[T]):
    """An object that describes the initialization of an instance.

    Initializer objects can be supplied as sole argument to a suitable __init__
    method to replace the usual processing of arguments with something else
    entirely.
    """

    def initialize(self, /, instance: T) -> None:
        raise MissingMethod(
            self,
            f"initialize an instance of type {type(instance)} with a"
        )

class Pixels(Encodable):
    """A container for non-negative values with uniform spacing.

    This class describes an abstract protocol for working with grayscale data.
    It supports working with individual values, vectors of values, images of
    values, videos of values, and stacks thereof.  Each pixel value is encoded
    as a discrete number of equidistant points.
    """

    def __new__(
            cls,
            data: npt.ArrayLike | Initializer[T],
            **kwargs) -> Pixels:
        # If someone attempts to instantiate the abstract pixels base class,
        # instantiate an appropriate subclass instead.
        if cls is Pixels:
            newcls = encoding(cls)
            assert newcls != cls # Avoid infinite recursion
            return newcls.__new__(newcls, data, **kwargs)
        # Otherwise, use the default __new__ method.
        return super().__new__(cls)

    def __init__(
            self,
            data: npt.ArrayLike | Initializer[Self],
            *,
            black: Real = 0,
            white: Real = 1,
            states: int = _default_states,
    ):
        """
        Initialize a Pixels container, based on the supplied arguments.

        Parameters
        ----------
        data: ArrayLike
            A real number, a nested sequence of real numbers, or an array of
            real numbers.
        black: real
            The number that is mapped to the intensity zero when viewing the data
            as a grayscale image.  Its default value is zero.
        white: real
            The number that is mapped to the intensity one when viewing the data
            as a grayscale image.  Its default value is one.
        states: int, optional
            The number of discrete states that the [floor, ceiling] interval is
             partitioned into.
        """
        super().__init__()
        if isinstance(data, Initializer):
            initializer = data
        else:
            discretization = Discretization((float(black), float(white)), (0, max(states, 1)-1))
            initializer = self._initializer_(data, discretization)
        initializer.initialize(self)

    @classmethod
    def _initializer_(
            cls: type[T],
            data: npt.ArrayLike,
            discretization: Discretization,
    ) -> Initializer[T]:
        """Create a suitable pixels initializer.

        Invoked in pixels' __init__ method to create an initializer object,
        which is then invoked to perform the actual initialization.  This
        double dispatch allows customization both across pixels classes (first
        invocation), and across data representations (second invocation).
        """
        _, _ = data, discretization
        raise MissingClassmethod(cls, "creating")

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        A tuple that describes the size of each axis.
        """
        ...

    @property
    @abstractmethod
    def discretization(self) -> Discretization:
        """
        The discretization between the [black, white] interval of pixel values to
        discrete integers.
        """
        ...

    @property
    def states(self) -> int:
        """
        The number of discrete states of each Pixel.
        """
        return self.discretization.states

    @property
    def eps(self) -> float:
        """
        The distance between any two adjacent Pixel states.
        """
        return self.discretization.eps

    @property
    @abstractmethod
    def data(self) -> npt.NDArray[np.float64]:
        """
        A read-only array of floats in [self.black, self.white] with the
        same shape as the pixels.
        """
        ...

    @property
    @abstractmethod
    def raw(self) -> npt.NDArray[uint]:
        """
        A read-only array of integers in [0, self.states-1] with the same
        shape as the pixels.
        """
        ...

    @property
    def black(self) -> float:
        """
        The smallest value that a pixel can hold.
        """
        return self.discretization.domain.lo

    @property
    def white(self) -> float:
        """
        The largest value that a pixel can hold.
        """
        return self.discretization.domain.hi

    @property
    def rank(self) -> int:
        """
        The number of axes of this container.
        """
        return len(self.shape)

    def __array__(self) -> npt.NDArray[np.float64]:
        return self.data

    def __len__(self: Pixels) -> int:
        """
        The size of the first axis of this container.
        """
        if len(self.shape) == 0:
            raise RuntimeError("A rank zero container has no length.")
        return self.shape[0]

    def __repr__(self) -> str:
        """
        A textual representation of this container.
        """
        name = type(self).__name__
        return f"{name}({self.data}, shape={self.shape}, discretization={self.discretization})"

    # Conversion from something to pixels.

    @classmethod
    def from_raw_file(
            cls,
            path: str | PathLike[str],
            shape: tuple[int, ...] | int,
            *,
            dtype: npt.DTypeLike | None = None,
            black: Real = 0,
            white: Real  = 1,
            states: int = _default_states) -> Pixels:
        # Ensure the path exists.
        path = canonicalize_path(path)
        if not path.exists():
            raise RuntimeError(f"The file {path} doesn't exist.")
        # Ensure the supplied shape is valid.
        shape = canonicalize_shape(shape)
        # Determine the raw file's element type.
        if dtype is None:
            size = prod(shape)
            fbytes = path.stat().st_size
            itemsize, mod = divmod(fbytes, size)
            if mod != 0:
                raise RuntimeError(
                    f"Raw file size {fbytes} is not divisible by the number of elements {size}."
                )
            dtype = f"u{itemsize}"
        dtype = np.dtype(dtype)
        black_default, white_default, states_default = dtype_black_white_states(dtype)
        discretization = Discretization((float(black), float(white)), (0, states-1))
        initializer = RawFilePixelsInitializer(
            path=path,
            temp=False,
            shape=shape,
            discretization=discretization,
            dtype=dtype
        )
        return RawFilePixels(initializer).encode_as(cls)

    # Conversion from pixels to something else.

    def to_raw_file(
            self,
            path: str | PathLike[str] | None = None,
            *,
            overwrite=True,
            temp: bool = False,
    ) -> RawFilePixels:
        raw_file = self.encode_as(RawFilePixels)
        if path is not None:
            raw_file.rename(path, overwrite=overwrite)
        return raw_file

    # getitem

    def __getitem__(self: Pixels, index: EllipsisType | int | slice | tuple[int | slice, ...]) -> Pixels:
        """
        Select a part of the supplied container.
        """
        return self._getitem_(canonicalize_index(index, self.shape))

    def _getitem_(self, index: tuple[int | slice, ...]) -> Pixels:
        _ = index
        raise MissingMethod(self, "indexing")

    # permute

    def permute(self, p0: int | tuple = (), /, *more: int) -> Pixels:
        """
        Reorder all axes according to the supplied axis numbers.
        """
        if isinstance(p0, tuple):
            permutation = p0 + more
        else:
            permutation = (p0,) + more
        nperm = len(permutation)
        rank = self.rank
        # Ensure that the permutation is well formed.
        if nperm > rank:
            raise ValueError(f"Invalid permutation {permutation} for data of rank {rank}.")
        for i, p in enumerate(permutation):
            if not isinstance(p, int):
                raise TypeError(f"The permutation entry {p} is not an integer.")
            if not 0 <= p < nperm:
                raise ValueError(f"Invalid entry {p} for permutation of length {nperm}.")
            if p in permutation[:i]:
                raise ValueError(f"Duplicate entry {p} in permutation {permutation}.")
        if nperm < rank:
            permutation += tuple(range(nperm, rank))
        # Call the actual implementation.
        cls = encoding(type(self))
        result = self.encode_as(cls)._permute_(permutation)
        # Ensure that the result has the expected shape.
        old_shape = self.shape
        new_shape = result.shape
        assert len(new_shape) == rank
        for i, p in enumerate(permutation):
            assert new_shape[i] == old_shape[p]
        for s1, s2 in zip(old_shape[nperm:], new_shape[nperm:]):
            assert s1 == s2
        return result

    def _permute_(self, permutation: tuple[int, ...]) -> Pixels:
        _ = permutation
        raise MissingMethod(self, "permuting")

    # align_with

    def align_with(
            self,
            *,
            black: Real | None = None,
            white: Real | None = None,
            states: int | None = None) -> Pixels:
        """Change the internal encoding of some Pixels to be aligned with the
        supplied discretization"""
        _black = self.black if black is None else float(black)
        _white = self.white if white is None else float(white)
        _states = self.states if states is None else states
        result = self._align_with_(_black, _white, _states)
        assert result.shape == self.shape
        assert result.states <= _states
        return result

    def _align_with_(self, black: float, white: float, states: int) -> Pixels:
        _, _, _ = black, white, states
        raise MissingMethod(self, "rescaling")

    # reshape

    def reshape(self, shape: tuple[int, ...]) -> Pixels:
        """
        Returns pixels with the original data and the supplied shape.
        """
        if prod(shape) != prod(self.shape):
            raise ValueError(f"Cannot reshape from shape {self.shape} to shape {shape}.")
        cls = encoding(type(self))
        result = self.encode_as(cls)._reshape_(shape)
        assert result.shape == shape
        assert result.discretization == self.discretization
        assert result.states == self.states
        return result

    def _reshape_(self, shape: tuple[int, ...]) -> Self:
        _ = shape
        raise MissingMethod(self, "reshaping")

    # broadcast_to

    def broadcast_to(self, shape: tuple[int, ...]) -> Pixels:
        """
        Replicate and stack the supplied data until it has the specified shape.
        """
        shape = canonicalize_shape(shape)
        result = self._broadcast_to_(shape)
        assert result.shape == shape
        assert result.discretization == self.discretization
        assert result.states == self.states
        return result

    def _broadcast_to_(self, shape: tuple[int, ...]) -> Pixels:
        _ = shape
        raise MissingMethod(self, "broadcasting")

    # predicates

    def __bool__(self) -> bool:
        raise RuntimeError("Never boolify Pixels, use .any() or .all() instead.")

    def any(self) -> bool:
        """
        Whether at least one pixel in the container has a non-black value.
        """
        cls = encoding(type(self))
        result = self.encode_as(cls)._any_()
        assert result is True or result is False
        return result

    def _any_(self) -> bool:
        raise MissingMethod(self, "testing for any non-black")

    def all(self) -> bool:
        """
        Whether all pixels in the container have a non-black value.
        """
        cls = encoding(type(self))
        result = self.encode_as(cls)._all_()
        assert result is True or result is False
        return result

    def _all_(self) -> bool:
        raise MissingMethod(self, "testing for all non-black")


    # and

    def __and__(self, other) -> Pixels:
        """
        The logical conjunction of the two supplied containers.

        The resulting container has two states: zero and one
        """
        a, b = broadcast(self, other)
        result = a._and_(b)
        assert result.shape == a.shape
        assert result.discretization == boolean_discretization
        return result

    def __rand__(self, other) -> Pixels:
        b, a = pixelize(self, other)
        return a.__and__(b)

    def _and_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "computing the logical conjunction of")

    # or

    def __or__(self, other) -> Pixels:
        """
        The logical disjunction of the two supplied containers.

        The resulting container has two states: zero and one
        """
        a, b = broadcast(self, other)
        result = a._or_(b)
        assert result.shape == a.shape
        assert result.discretization == boolean_discretization
        return result

    def __ror__(self, other) -> Pixels:
        b, a = pixelize(self, other)
        return a.__or__(b)

    def _or_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "computing the logical disjunction of")

    # xor

    def __xor__(self, other) -> Pixels:
        """
        The exclusive disjunction of the two supplied containers.

        The resulting container has two states: zero and one
        """
        a, b = broadcast(self, other)
        result = a._xor_(b)
        assert result.shape == a.shape
        assert result.discretization == boolean_discretization
        return result

    def __rxor__(self, other) -> Pixels:
        b, a = pixelize(self, other)
        return a.__xor__(b)

    def _xor_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "logical xor-ing")

    # lshift

    def __lshift__(self, amount: int) -> Pixels:
        """
        Multiply each value by two to the power of the supplied amount.
        """
        if amount >= 0:
            return self * 2**amount
        else:
            return self >> (-amount)

    # rshift

    def __rshift__(self, amount: int) -> Pixels:
        """
        Divide each value by two to the power of the supplied amount.
        """
        if amount >= 0:
            return self / 2**amount
        else:
            return self << (-amount)

    # abs

    def __abs__(self) -> Pixels:
        """
        Negate each pixel value less than zero.
        """
        cls = encoding(type(self))
        result = self.encode_as(cls)._abs_()
        assert result.shape == self.shape
        sdomain = self.discretization.domain
        rdomain = result.discretization.domain
        assert rdomain.lo == max(0.0, sdomain.lo)
        assert rdomain.hi == max(abs(sdomain.lo), abs(sdomain.hi))
        return result

    def _abs_(self) -> Self:
        raise MissingMethod(self, "computing the absolute of")

    # invert

    def __invert__(self) -> Pixels:
        """
        Flip the sign of each pixel.
        """
        cls = encoding(type(self))
        result = self.encode_as(cls)._invert_()
        assert result.shape == self.shape
        assert result.states == self.states
        return result

    def _invert_(self) -> Self:
        raise MissingMethod(self, "inverting")

    # neg

    __neg__ = __invert__

    # pos

    def __pos__(self) -> Pixels:
        """
        Do nothing.
        """
        cls = encoding(type(self))
        result = self.encode_as(cls)
        assert result.shape == self.shape
        assert result.discretization == self.discretization
        assert result.states == self.states
        return result

    # add

    def __add__(self, other):
        """
        Add the values of the two containers.
        """
        a, b = broadcast(self, other)
        result = a._add_(b)
        assert result.shape == a.shape
        return result

    def __radd__(self, other):
        b, a = pixelize(self, other)
        return a.__add__(b)

    def _add_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "adding")

    # sub

    def __sub__(self, other):
        """
        Subtract the values of the two containers.
        """
        a, b = broadcast(self, other)
        result = a._sub_(b)
        pass # TODO

    def __rsub__(self, other):
        b, a = pixelize(self, other)
        return a.__sub__(b)

    def _sub_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "subtracting")

    # mul

    def __mul__(self, other) -> Pixels:
        """
        Multiply the values of the containers.
        """
        a, b = broadcast(self, other)
        result = a._mul_(b)
        return result # TODO

    def __rmul__(self, other):
        b, a = pixelize(self, other)
        return a.__mul__(b)

    def _mul_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "multiplying")

    # pow

    def __pow__(self, exponent) -> Pixels:
        """
        Raise each value to the specified power.
        """
        a, b = broadcast(self, exponent)
        result = a._pow_(b)
        return result # TODO

    def _pow_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "exponentiating")

    # truediv

    def __truediv__(self, other) -> Pixels:
        """
        Divide the values of the two containers.
        """
        a, b = broadcast(self, other)
        result = a._truediv_(b)
        return result # TODO

    def _truediv_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "dividing")

    # mod

    def __mod__(self, other) -> Pixels:
        """
        Left value modulo right value.
        """
        a, b = broadcast(self, other)
        result = a._mod_(b)
        return result # TODO

    def __rmod__(self, other) -> Pixels:
        b, a = pixelize(self, other)
        return a.__mod__(b)

    def _mod_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "computing the modulus of")

    # floordiv

    def __floordiv__(self, other) -> Pixels:
        """
        Divide the values of the two containers and round the result down to the next integer.
        """
        a, b = broadcast(self, other)
        result = a._floordiv_(b)
        return result # TODO

    def __rfloordiv__(self, other) -> Pixels:
        b, a = pixelize(self, other)
        return a.__floordiv__(b)

    def _floordiv_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "floor-dividing")

    # lt

    def __lt__(self, other) -> Pixels:
        """
        One wherever the left value is smaller than the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        result = a._lt_(b)
        assert result.shape == a.shape
        assert result.discretization == boolean_discretization
        return result

    def _lt_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "comparing")

    # gt

    def __gt__(self, other) -> Pixels:
        """
        One wherever the left value is greater than the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        result = a._gt_(b)
        assert result.shape == a.shape
        assert result.discretization == boolean_discretization
        return result

    def _gt_(self: Self, other: Self) -> Self:
        return other._lt_(self)

    # le

    def __le__(self, other) -> Pixels:
        """
        One wherever the left value is less than or equal to the right, zero
        otherwise.
        """
        a, b = broadcast(self, other)
        result = a._le_(b)
        assert result.shape == a.shape
        assert result.discretization == boolean_discretization
        return result

    def _le_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "comparing")

    # ge

    def __ge__(self, other) -> Pixels:
        """
        One wherever the left value is greater than or equal to the right, zero
        otherwise.
        """
        a, b = broadcast(self, other)
        result = a._ge_(b)
        assert result.shape == a.shape
        assert result.discretization == boolean_discretization
        return result

    def _ge_(self: Self, other: Self) -> Self:
        return other._le_(self)

    # eq

    def __eq__(self, other) -> Pixels: # type: ignore
        """
        One wherever the left value is equal to the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        result = a._eq_(b)
        assert result.shape == a.shape
        assert result.discretization == boolean_discretization
        return result

    def _eq_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "determining the equality of")

    # ne

    def __ne__(self, other) -> Pixels: # type: ignore
        """
        One wherever the left value is different than the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        result = a._ne_(b)
        assert result.shape == a.shape
        assert result.discretization == boolean_discretization
        return result

    def _ne_(self: Self, other: Self) -> Self:
        _ = other
        return (self._eq_(other))._invert_()

    # sum

    def sum(self, axis: int | tuple[int, ...]=0, keepdims: bool=False) -> Pixels:
        """
        The sum of all values along the specified axis or axes.
        """
        rank = self.rank
        axes = canonicalize_axes(axis, rank)
        window_sizes = tuple((size if axis in axes else 1) for axis, size in enumerate(self.shape))
        result = self.rolling_sum(window_sizes)
        if keepdims:
            return result
        else:
            shape = tuple(size for axis, size in enumerate(self.shape) if axis not in axes)
            return result.reshape(shape)

    def rolling_sum(self, window_size: int | tuple[int, ...]) -> Pixels:
        """
        The rolling sum for a given window size.
        """
        window_sizes = canonicalize_window_sizes(window_size, self.shape)
        cls = encoding(type(self))
        result = self.encode_as(cls)._rolling_sum_(window_sizes)
        assert result.shape == tuple((s - w + 1) for s, w in zip(self.shape, window_sizes))
        return result

    def _rolling_sum_(self, window_sizes: tuple[int, ...]) -> Self:
        _ = window_sizes
        raise MissingMethod(self, "computing the sum of")

    # average

    def average(self, axis: int | tuple[int, ...]=0, keepdims: bool=False) -> Pixels:
        """
        The average of all values along the specified axis or axes.
        """
        rank = self.rank
        axes = canonicalize_axes(axis, rank)
        window_sizes = tuple((size if axis in axes else 1) for axis, size in enumerate(self.shape))
        result = self.rolling_average(window_sizes)
        if keepdims:
            return result
        else:
            shape = tuple(size for axis, size in enumerate(self.shape) if axis not in axes)
            return result.reshape(shape)

    def rolling_average(self, window_size: int | tuple[int, ...]) -> Pixels:
        """
        The rolling average for a given window size.
        """
        window_sizes = canonicalize_window_sizes(window_size, self.shape)
        amount = prod(window_sizes)
        return self.rolling_sum(window_sizes) / amount

    # median

    def median(self, axis: int | tuple[int, ...]=0, keepdims: bool=False) -> Pixels:
        """
        The median of all values along the specified axis or axes.
        """
        rank = self.rank
        axes = canonicalize_axes(axis, rank)
        window_sizes = tuple((size if axis in axes else 1) for axis, size in enumerate(self.shape))
        result = self.rolling_median(window_sizes)
        if keepdims:
            return result
        else:
            shape = tuple(size for axis, size in enumerate(self.shape) if axis not in axes)
            return result.reshape(shape)

    def rolling_median(self, window_size: int | tuple[int, ...]) -> Pixels:
        window_sizes = canonicalize_window_sizes(window_size, self.shape)
        cls = encoding(type(self))
        result = self.encode_as(cls)._rolling_median_(window_sizes)
        assert result.shape == tuple((s - w + 1) for s, w in zip(self.shape, window_sizes))
        return result

    def _rolling_median_(self, window_sizes: tuple[int, ...]) -> Self:
        _ = window_sizes
        raise MissingMethod(self, "computing the median of")

    # TODO new methods: variance, convolve, fft


P = TypeVar('P', bound=Pixels)


@dataclass(frozen=True)
class PixelsInitializer(Initializer[P]):
    pass


###############################################################################
###
### Concrete Pixels

class ConcretePixels(Pixels):
    _shape: tuple[int, ...]
    _discretization: Discretization

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def discretization(self) -> Discretization:
        return self._discretization


CP = TypeVar('CP', bound=ConcretePixels)


@dataclass(frozen=True)
class ConcretePixelsInitializer(PixelsInitializer[CP]):
    shape: tuple[int, ...]
    discretization: Discretization

    def initialize(self, /, instance: CP):
        instance._shape = self.shape
        instance._discretization = self.discretization


###############################################################################
###
### File Pixels

class FilePixels(ConcretePixels):
    """Pixels stored in a file.

    The Pixels that are stored in a file don't reside in main memory, but
    somewhere in the file system.  The file can either be temporary, in which
    case it is removed once its corresponding class is no longer reachable, or
    it can be persistent.  A persistent file can be used for storing pixels
    across sessions, for archival, or for communicating it with others.

    """
    # When a temporary file object is deleted, a finalizer will also remove the
    # actual file associated with it.  However, both the file's path and
    # whether it is temporary can change over time, and the finalizer has to
    # keep a reference to both without accidentally keeping the file object
    # alive.  To achieve this, we store both properties in one-element lists,
    # and pass those two lists to the finalizer.
    _path_cell: list[Path]
    _temp_cell: list[bool]

    @property
    def path(self) -> Path:
        return self._path_cell[0]

    @property
    def is_temporary(self) -> bool:
        return self._temp_cell[0]

    def rename(
            self,
            path: str | PathLike[str],
            *,
            overwrite: bool = False,
            temporary: bool = False) -> Self:
        path = canonicalize_path(path)
        if path.exists():
            if path.is_dir():
                raise RuntimeError(f"Cannot overwrite existing directory {path}.")
            if not path.is_file():
                raise RuntimeError(f"Cannot overwrite non-file {path}.")
            elif overwrite:
                path.unlink()
            else:
                raise RuntimeError(f"The file {path} already exists.")
        self._path_cell[0] = self.path.rename(path)
        self._temp_cell[0] = temporary
        return self


FP = TypeVar('FP', bound=FilePixels)


@dataclass(frozen=True)
class FilePixelsInitializer(ConcretePixelsInitializer[FP]):
    path: str | PathLike[str]
    temp: bool

    def initialize(self, /, instance: FP):
        super().initialize(instance)
        abspath = Path(self.path).expanduser().resolve(strict=True)
        instance._path_cell = [abspath]
        instance._temp_cell = [self.temp]


class RawFilePixels(FilePixels):
    _dtype: npt.DTypeLike


RP = TypeVar('RP', bound=RawFilePixels)


@dataclass(frozen=True)
class RawFilePixelsInitializer(FilePixelsInitializer[RP]):
    """Turn an existing raw file into pixels."""
    dtype: npt.DTypeLike

    def initialize(self, /, instance: RP):
        super().initialize(instance)
        instance._dtype = self.dtype


###############################################################################
###
### Auxiliary Functions


def MissingMethod(self, action) -> TypeError:
    return TypeError(f"No method for {action} objects of type {type(self)}.")


def MissingClassmethod(cls, action) -> TypeError:
    return TypeError(f"No classmethod for {action} of type {cls}.")


def canonicalize_path(path) -> Path:
    return Path(path).expanduser().absolute()


def canonicalize_shape(shape) -> tuple[int, ...]:
    if isinstance(shape, int):
        dims = (shape,)
    else:
        dims = tuple(int(x) for x in shape)
        return dims
    for dim in dims:
        if dim < 0:
            raise TypeError(f"Not a valid shape dimension: {dim}")
    return dims


def canonicalize_index(index, shape: tuple) -> tuple[int | slice, ...]:
    # Step 1 - Convert the index into a tuple.
    ituple : tuple
    if index == ...:
        ituple = tuple(slice(None) for _ in shape)
    elif isinstance(index, int) or isinstance(index, slice):
        ituple = (index,)
    elif isinstance(index, tuple):
        ituple = index
    else:
        raise TypeError(f"Invalid index {index}.")
    # Step 2 - Ensure that ituple and shape have the same length.
    ni = len(ituple)
    ns = len(shape)
    if ni < ns:
        ituple += tuple(slice(None) for _ in range(ns - ni))
    elif ni > ns:
        raise ValueError(f"Too many indices for shape {shape}.")
    # Step 3 - Ensure that all entries are well formed.
    for entry, size in zip(ituple, shape):
        if isinstance(entry, int):
            if not (-size <= entry < size):
                raise IndexError(f"Index {entry} out of range for axis with size {size}.")
        elif isinstance(entry, slice):
            # Python treats out-of-bounds slices as empty, so any slice is
            # valid for any size.
            pass
        else:
            raise TypeError(f"Invalid index component {entry}.")
    return ituple


def canonicalize_axes(
        axis: int | tuple[int, ...],
        rank: int) -> tuple[int, ...]:
    assert 0 <= rank
    if isinstance(axis, int):
        axes = (axis,)
    elif isinstance(axis, tuple):
        axes = axis
    else:
        raise TypeError(f"Invalid axis specifier {axis}.")
    for index, axis in enumerate(axes):
        if not (0 <= axis < rank):
            raise ValueError(f"Invalid axis {axis} for data of rank {rank}.")
        if axis in axes[:index]:
            raise ValueError(f"Duplicate axis {axis} in {axes}.")
    return axes


def canonicalize_window_sizes(
        window_size: int | tuple[int, ...],
        shape: tuple[int, ...]) -> tuple[int, ...]:
    rank = len(shape)
    if isinstance(window_size, int):
        window_sizes = (window_size,) + (1,) * max(0, rank-1)
    elif isinstance(window_size, tuple):
        n = len(window_size)
        window_sizes = window_size + (1,) * max(0, rank-n)
    else:
        raise TypeError(f"Invalid window size specifier {window_size}.")
    if len(window_sizes) > rank:
        raise ValueError(f"Too many window sizes for data of rank {rank}.")
    for ws, size in zip(window_sizes, shape):
        if (not isinstance(ws, int)) or ws < 1:
            raise ValueError(f"Invalid window size {ws}.")
        if ws > size:
            raise ValueError(f"Too large window size {ws} for axis of size {size}.")
    return window_sizes


def pixelize(a: Pixels | Real, b: Pixels | Real) -> tuple[Pixels, Pixels]:
    """
    Coerce the two arguments to the same class.
    """
    if isinstance(a, Pixels):
        if isinstance(b, Pixels):
            # Handle the case where a and b are Pixels objects.
            cls = encoding(type(a), type(b))
            pxa = a.encode_as(cls)
            pxb = b.encode_as(cls)
        else:
            # Handle the case where a is a Pixels object and b is a Real.
            cls = encoding(type(a))
            pxa = a.encode_as(cls)
            val = float(b)
            pxb = cls(val, black=val, white=val, states=1)
    else:
        if isinstance(b, Pixels):
            # Handle the case where a is a Real and b is a Pixels object.
            cls = encoding(type(b))
            pxb = b.encode_as(cls)
            val = float(a)
            pxa = cls(val, black=val, white=val, states=1)
        else:
            # Handle the case where a and b are Real numbers.
            cls = encoding()
            vala = float(a)
            valb = float(b)
            pxa = cls(vala, black=vala, white=vala, states=1)
            pxb = cls(valb, black=valb, white=valb, states=1)
    return pxa, pxb


def align(a: Pixels | Real, b: Pixels | Real) -> tuple[Pixels, Pixels]:
    """
    Coerce the two arguments to have the same class and a compatible encoding.
    """
    pxa, pxb = pixelize(a, b)
    # Ensure both Pixels have a compatible encoding.
    black = min(pxa.black, pxb.black)
    white = max(pxa.white, pxb.white)
    eps = (white - black) / _default_states
    if pxa.states == 1 and pxb.states == 1:
        if black == white:
            states = 1
        else:
            states = 2
    elif pxa.states > 1 and pxb.states == 1:
        states = ceil((white - black) / min(eps, pxa.eps)) + 1
    elif pxa.states == 1 and pxb.states > 1:
        states = ceil((white - black) / min(eps, pxb.eps)) + 1
    else:
        states = ceil((white - black) / min(eps, pxa.eps, pxb.eps)) + 1
    pxa = pxa.align_with(black=black, white=white, states=states)
    pxb = pxb.align_with(black=black, white=white, states=states)
    return pxa, pxb


def broadcast_shapes(shape1: tuple, shape2: tuple) -> tuple:
    """Broadcast the two supplied shapes or raise an error."""
    rank1 = len(shape1)
    rank2 = len(shape2)
    axes = []
    minrank = min(rank1, rank2)
    for axis in range(minrank):
        d1, d2 = shape1[axis], shape2[axis]
        if d1 == 1:
            axes.append(d2)
        elif d2 == 1:
            axes.append(d1)
        elif d1 == d2:
            axes.append(d1)
        else:
            raise ValueError(f"Size mismatch in axis {axis}.")
    if rank1 < rank2:
        return tuple(axes) + shape2[minrank:]
    else:
        return tuple(axes) + shape1[minrank:]


def broadcast(a: Pixels | Real, b: Pixels | Real) -> tuple[Pixels, Pixels]:
    pxa, pxb = align(a, b)
    shape = broadcast_shapes(pxa.shape, pxb.shape)
    return pxa.broadcast_to(shape), pxb.broadcast_to(shape)

def dtype_black_white_states(dtype: npt.DTypeLike) -> tuple[Real, Real, int]:
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return (0.0, 1.0, 2**53)
    if dtype == np.float32:
        return (0.0, 1.0, 2**24)
    elif dtype.kind == 'u':
        nbits = dtype.itemsize * 8
        return (0, 2**nbits-1, 2**nbits)
    elif dtype.kind == 'i':
        nbits = dtype.itemsize * 8
        return (-(2**(nbits-1)), (2**(nbits-1))-1, 2**nbits)
    else:
        raise TypeError(f"Cannot convert {dtype} objects to pixels values.")
