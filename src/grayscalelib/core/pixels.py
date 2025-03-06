from __future__ import annotations

import operator
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from math import floor, prod
from os import PathLike
from pathlib import Path
from sys import float_info
from typing import Callable, Generic, Protocol, Self, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

from grayscalelib.core.discretization import Discretization, boolean_discretization
from grayscalelib.core.encodable import Encodable, choose_encoding

###############################################################################
###
### Global Variables


_enforced_pixels_type: type[Pixels] | None = None

_default_pixels_type: type[Pixels] | None = None

_default_states: int = 256

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


@runtime_checkable
class Real(Protocol):
    def __float__(self) -> float: ...


T = TypeVar("T")


@dataclass(frozen=True)
class Initializer(Generic[T]):
    """An object that describes the initialization of an instance.

    Initializer objects can be supplied as sole argument to a suitable __init__
    method to replace the usual processing of arguments with something else
    entirely.
    """

    def initialize(self, /, instance: T) -> None:
        raise MissingMethod(instance, "initializing")


###############################################################################
###
### The Pixels Class


class Pixels(Encodable):
    """A container for non-negative values with uniform spacing.

    This class describes an abstract protocol for working with grayscale data.
    It supports working with individual values, vectors of values, images of
    values, videos of values, and stacks thereof.  Each pixel value is encoded
    as a discrete number of equidistant points.
    """

    def __new__(cls, data: npt.ArrayLike | Initializer[T], **kwargs) -> Pixels:
        # If someone attempts to instantiate the abstract pixels base class,
        # instantiate an appropriate subclass instead.
        if cls is Pixels:
            newcls = encoding(cls)
            assert newcls != cls  # Avoid infinite recursion
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
            discretization = Discretization((float(black), float(white)), (0, max(states, 1) - 1))
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
    def roundoff(self) -> float:
        """
        The maximum error introduced any operation on these Pixels.
        """
        return 0.5 * self.eps + float_info.epsilon

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
        return (
            f"{name}({self.data}, black={self.black}, white={self.white}, states={self.states})"
        )

    # Conversion from raw files to pixels.

    @classmethod
    def from_raw_file(
        cls,
        path: str | PathLike[str],
        shape: tuple[int, ...] | int,
        *,
        dtype: npt.DTypeLike | None = None,
        black: Real | None = None,
        white: Real | None = None,
        states: int | None = None,
    ) -> Pixels:
        # Ensure the path exists.
        path = canonicalize_path(path)
        if not path.exists():
            raise RuntimeError(f"The file {path} doesn't exist.")
        if not path.is_file():
            raise RuntimeError(f"Not a file: {path}.")
        # Ensure the supplied shape is valid.
        shape = canonicalize_shape(shape)
        count = prod(shape)
        # Determine the raw file's element type.
        if dtype is None:
            fbytes = path.stat().st_size
            itemsize, mod = divmod(fbytes, count)
            if mod != 0:
                raise RuntimeError(
                    f"Raw file size {fbytes} is not divisible by the number of elements {count}."
                )
            dtype = f"u{itemsize}"
        dtype = np.dtype(dtype)
        black_default, white_default, states_default = dtype_black_white_states(dtype)
        _black = black_default if black is None else black
        _white = white_default if white is None else white
        _states = states_default if states is None else states
        # Create a suitable Pixels object.
        data = np.fromfile(path, dtype, count).reshape(shape)
        return cls(data, black=_black, white=_white, states=_states)

    # Conversion from pixels to raw files.

    def to_raw_file(
        self,
        path: str | PathLike[str] | None = None,
        *,
        overwrite=True,
    ) -> None:
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
        self.raw.tofile(path)

    # getitem

    def __getitem__(self: Pixels, index) -> Pixels:
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
        for s1, s2 in zip(old_shape[nperm:], new_shape[nperm:], strict=False):
            assert s1 == s2
        return result

    def _permute_(self, permutation: tuple[int, ...]) -> Pixels:
        _ = permutation
        raise MissingMethod(self, "permuting")

    # rediscretize

    def rediscretize(self, discretization: Discretization):
        cls = encoding(type(self))
        result = self.encode_as(cls)._rediscretize_(discretization)
        assert result.discretization == discretization
        assert result.shape == self.shape
        return result

    def _rediscretize_(self, dr) -> Self:
        _ = dr
        raise MissingMethod(self, "rediscretizing")

    # reencode

    def reencode(
        self,
        *,
        cls: type[Pixels] | None = None,
        black: Real | None = None,
        white: Real | None = None,
        states: int | None = None,
    ) -> Pixels:
        """Replace each pixel with the closest value in the partitioning of
        [black, white] into the supplied number of equidistant states.

        Parameters
        ----------
        self: Pixels
            The pixels being reencoded.
        cls: type[Pixels]
            The class of the resulting container.
        black: Real or None
            The value corresponding to minimum intensity.
        white: Real or None
            The value corresponding to maximum intensity.
        states: int or None
            The number of discrete, equidistant states of the space being
            aligned with.

        Returns
        -------
        Pixels
            A container with the same shape as the supplied one, but possibly
            a different discretization.
        """
        _cls = type(self) if cls is None else cls
        _black = self.black if black is None else float(black)
        _white = self.white if white is None else float(white)
        _states = self.states if states is None else states
        if _white < _black:
            _black, _white = _white, _black
        if _states < 1:
            raise TypeError("Discretization requires at least one state.")
        if not (_black <= self.black <= self.white <= _white):
            raise TypeError("Bounds must encompass the full pixel range.")
        try:
            result = self.encode_as(_cls)._reencode_(_black, _white, _states)
        except MissingMethod:
            result = self._reencode_(_black, _white, _states).encode_as(_cls)
        assert result.shape == self.shape
        assert result.states <= _states
        return result

    def _reencode_(self, black: float, white: float, states: int) -> Pixels:
        _, _, _ = black, white, states
        raise MissingMethod(self, "reencoding")

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
        Flip all values such that the black and white spectrum is reversed.
        """
        cls = encoding(type(self))
        px = self.encode_as(cls)
        pxd = px.discretization
        dr = Discretization(pxd.domain, pxd.codomain, not pxd.flip)
        return px.rediscretize(dr)

    # neg

    def __neg__(self) -> Pixels:
        """
        Flip all values such that the black and white spectrum is reversed.
        """
        cls = encoding(type(self))
        px = self.encode_as(cls)
        pxd = px.discretization
        dr = Discretization((-pxd.domain.lo, -pxd.domain.hi), pxd.codomain, pxd.flip)
        return px._rediscretize_(dr)

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
        # Compute the resulting discretization
        black = a.black + b.black
        white = a.white + b.white
        if black == white:
            return type(a)(black, black=black, white=white, states=1).broadcast_to(a.shape)
        if a.states == 1:
            bd = b.discretization
            dr = Discretization((black, white), bd.codomain, bd.flip)
            return b.rediscretize(dr)
        if b.states == 1:
            ad = a.discretization
            dr = Discretization((black, white), ad.codomain, ad.flip)
            return a.rediscretize(dr)
        states = round((white - black) / min(a.eps, b.eps)) + 1
        dr = Discretization((black, white), (0, states - 1))
        # Compute the result.
        result = a._add_(b, dr)
        assert result.shape == a.shape
        assert result.discretization == dr
        return result

    def __radd__(self, other):
        b, a = pixelize(self, other)
        return a.__add__(b)

    def _add_(self: Self, other: Self, dr: Discretization) -> Self:
        _, _ = other, dr
        raise MissingMethod(self, "adding")

    # sub

    def __sub__(self, other):
        """
        Subtract the values of the two containers.
        """
        a, b = broadcast(self, other)
        return a.__add__(-b)

    def __rsub__(self, other):
        b, a = pixelize(self, other)
        return a.__add__(-b)

    # mul

    def __mul__(self, other) -> Pixels:
        """
        Multiply the values of the containers.
        """
        a, b = broadcast(self, other)
        # Compute the resulting discretization
        x1, x2 = a.black, a.white
        y1, y2 = b.black, b.white
        f1, f2, f3, f4 = x1 * y1, x1 * y2, x2 * y1, x2 * y2
        black = min(f1, f2, f3, f4)
        white = max(f1, f2, f3, f4)
        if black == white:
            return type(a)(black, black=black, white=white, states=1).broadcast_to(self.shape)
        if a.states == 1:
            bd = b.discretization
            flip = bd.flip ^ (a.black < 0)
            dr = Discretization((black, white), bd.codomain, flip)
            return b.rediscretize(dr)
        if b.states == 1:
            ad = a.discretization
            flip = ad.flip ^ (b.black < 0)
            dr = Discretization((black, white), ad.codomain, flip)
            return a.rediscretize(dr)
        states = a.states * b.states
        dr = Discretization((black, white), (0, states - 1))
        # Compute the result.
        result = a._mul_(b, dr)
        assert result.shape == a.shape
        assert result.discretization == dr
        return result

    def __rmul__(self, other):
        b, a = pixelize(self, other)
        return a.__mul__(b)

    def _mul_(self: Self, other: Self, dr: Discretization) -> Self:
        _, _ = other, dr
        raise MissingMethod(self, "multiplying")

    # pow

    def __pow__(self, exponent: Real) -> Pixels:
        """
        Raise each value to the specified power.
        """
        cls = encoding(type(self))
        base = self.encode_as(cls)
        # Compute the resulting discretization
        result = base._pow_(float(exponent))
        assert result.shape == base.shape
        return result

    def _pow_(self, exponent: float) -> Self:
        _ = exponent
        raise MissingMethod(self, "exponentiating")

    # truediv

    def __truediv__(self, other) -> Pixels:
        """
        Divide the values of the two containers.
        """
        a, b = broadcast(self, other)
        # Handle the special case where b has only a single state.
        if b.states == 1:
            if b.black == 0:
                value = (self.white + self.black) / 2
                return type(self)(value, black=value, white=value, states=1)
            else:
                return a * (1 / b.black)
        # Derive the interval boundaries of 1/b, while excluding any values
        # that are within (-b.eps, +b.eps) so that we avoid infinities.
        if 0 <= b.black:
            binv_hi = 1 / max(+b.eps, b.black)
            binv_lo = 1 / max(+b.eps, b.white)
        elif b.white <= 0:
            binv_lo = 1 / min(-b.eps, b.white)
            binv_hi = 1 / min(-b.eps, b.black)
        else:
            binv_lo = 1 / -b.eps
            binv_hi = 1 / +b.eps
        assert binv_lo <= binv_hi
        # Determine the boundaries of a * [binv_lo, binv_hi].
        x1, x2 = a.black, a.white
        y1, y2 = binv_lo, binv_hi
        f1, f2, f3, f4 = x1 * y1, x1 * y2, x2 * y1, x2 * y2
        black = min(f1, f2, f3, f4)
        white = max(f1, f2, f3, f4)
        # Determine the resulting discretization.
        states = a.states * b.states
        dr = Discretization((black, white), (0, states - 1))
        # Chose the value to assign to 0 / 0
        nan = (black + white) / 2
        # Perform the actual division
        result = a._truediv_(b, dr, nan)
        assert result.shape == a.shape
        return result

    def __rtruediv__(self, other):
        b, a = pixelize(self, other)
        return a.__truediv__(b)

    def _truediv_(self: Self, other: Self, dr: Discretization, nan: float) -> Self:
        _, _, _ = other, dr, nan
        raise MissingMethod(self, "dividing")

    # mod

    def __mod__(self, other) -> Pixels:
        """
        Left value modulo right value.
        """
        a, b = broadcast(self, other)
        result = a._mod_(b)
        return result  # TODO

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
        # Handle the special case where b has only a single state.
        if b.states == 1 and b.black == 0:
            value = (self.white + self.black) // 2
            return type(self)(value, black=value, white=value, states=1)
        # Derive the interval boundaries of 1/b, while excluding any values
        # that are within (-b.eps, +b.eps) so that we avoid infinities.
        if 0 <= b.black:
            binv_hi = 1 / max(+b.eps, b.black)
            binv_lo = 1 / max(+b.eps, b.white)
        elif b.white <= 0:
            binv_lo = 1 / min(-b.eps, b.white)
            binv_hi = 1 / min(-b.eps, b.black)
        else:
            binv_lo = 1 / -b.eps
            binv_hi = 1 / +b.eps
        assert binv_lo <= binv_hi
        # Determine the boundaries of a * [binv_lo, binv_hi].
        x1, x2 = a.black, a.white
        y1, y2 = binv_lo, binv_hi
        f1, f2, f3, f4 = x1 * y1, x1 * y2, x2 * y1, x2 * y2
        black = floor(min(f1, f2, f3, f4))
        white = floor(max(f1, f2, f3, f4))
        print(f"{binv_lo=} {binv_hi=} {x1=} {x2=} {y1=} {y2=} {black=} {white=}")
        # Determine the intermediate discretization and the resulting discretization.
        states = int(white - black) + 1
        delta = 0.5
        di = Discretization((black + delta, white + delta), (0, states - 1))
        dr = Discretization((black, white), (0, states - 1))
        # Chose the value to assign to 0 / 0
        nan = (black + white) / 2 + delta
        # Perform the actual division
        result = a._truediv_(b, di, nan).rediscretize(dr)
        assert result.shape == a.shape
        return result

    def __rfloordiv__(self, other) -> Pixels:
        b, a = pixelize(self, other)
        return a.__floordiv__(b)

    # comparisons

    def __lt__(self, other) -> Pixels:
        """
        One wherever the left value is smaller than the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        # Handle the case where [a.black, a.white] < [b.black, b.white]
        if a.white < b.black:
            return type(a)(1, black=1, white=1, states=1).broadcast_to(a.shape)
        # Handle the case where [b.black, b.white] <= [a.black, a.white]
        if b.white <= a.black:
            return type(a)(0, black=0, white=0, states=1).broadcast_to(a.shape)
        # Ensure that the first operand has more than one state.
        if a.states == 1:
            result = b._cmp_(a, operator.gt)
        else:
            result = a._cmp_(b, operator.lt)
        assert result.shape == a.shape
        return result

    def __gt__(self, other) -> Pixels:
        """
        One wherever the left value is greater than the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        # Handle the case where [a.black, a.white] <= [b.black, b.white]
        if a.white <= b.black:
            return type(a)(0, black=0, white=0, states=1).broadcast_to(a.shape)
        # Handle the case where [b.black, b.white] < [a.black, a.white]
        if b.white <= a.black:
            return type(a)(1, black=1, white=1, states=1).broadcast_to(a.shape)
        # Ensure that the first operand has more than one state.
        if a.states == 1:
            result = b._cmp_(a, operator.lt)
        else:
            result = a._cmp_(b, operator.gt)
        assert result.shape == a.shape
        return result

    def __le__(self, other) -> Pixels:
        """
        One wherever the left value is smaller than or equal to the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        # Handle the case where [a.black, a.white] <= [b.black, b.white]
        if a.white <= b.black:
            return type(a)(1, black=1, white=1, states=1).broadcast_to(a.shape)
        # Handle the case where [b.black, b.white] < [a.black, a.white]
        if b.white < a.black:
            return type(a)(0, black=0, white=0, states=1).broadcast_to(a.shape)
        # Ensure that the first operand has more than one state.
        if a.states == 1:
            result = b._cmp_(a, operator.ge)
        else:
            result = a._cmp_(b, operator.le)
        assert result.shape == a.shape
        return result

    def __ge__(self, other) -> Pixels:
        """
        One wherever the left value is greater than or equal to the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        # Handle the case where [a.black, a.white] < [b.black, b.white]
        if a.white < b.black:
            return type(a)(0, black=0, white=0, states=1).broadcast_to(a.shape)
        # Handle the case where [b.black, b.white] <= [a.black, a.white]
        if b.white <= a.black:
            return type(a)(1, black=1, white=1, states=1).broadcast_to(a.shape)
        # Ensure that the first operand has more than one state.
        if a.states == 1:
            result = b._cmp_(a, operator.le)
        else:
            result = a._cmp_(b, operator.ge)
        assert result.shape == a.shape
        return result

    def __eq__(self, other) -> Pixels:  # type: ignore
        """
        One wherever the left value is equal to the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        # Handle the case of identical one-element domains.
        if a.black == a.white == b.black == b.white:
            return type(a)(1, black=1, white=1, states=1).broadcast_to(a.shape)
        # Handle the case of completely disjoint domains.
        if (a.white < b.black) or (b.white < a.black):
            return type(a)(0, black=0, white=0, states=1).broadcast_to(a.shape)
        # Ensure that the first operand has more than one state.
        if a.states == 1:
            result = b._cmp_(a, operator.eq)
        else:
            result = a._cmp_(b, operator.eq)
        assert result.shape == a.shape
        return result

    def __ne__(self, other) -> Pixels:  # type: ignore
        """
        One wherever the left value differs from the right, zero otherwise.
        """
        a, b = broadcast(self, other)
        # Handle the case of identical one-element domains.
        if a.black == a.white == b.black == b.white:
            return type(a)(0, black=0, white=0, states=1).broadcast_to(a.shape)
        # Handle the case of completely disjoint domains.
        if (a.white < b.black) or (b.white < a.black):
            return type(a)(1, black=1, white=1, states=1).broadcast_to(a.shape)
        # Ensure that the first operand has more than one state.
        if a.states == 1:
            result = b._cmp_(a, operator.ne)
        else:
            result = a._cmp_(b, operator.ne)
        assert result.shape == a.shape
        return result

    def _cmp_(self: Self, other: Self, op: Callable[[float, float], bool]) -> Self:
        """Returns Pixels that are one where the supplied comparison holds, and
        zero otherwise.

        A caller must ensure that the first two arguments are Pixels of the
        same class, and that the first argument has more than one state.
        """
        _, _ = other, op
        raise MissingMethod(self, "comparing")

    # sum

    def sum(self, axis: int | tuple[int, ...] = 0, keepdims: bool = False) -> Pixels:
        """
        The sum of all values along the specified axis or axes.
        """
        rank = self.rank
        axes = canonicalize_axes(axis, rank)
        window_sizes = tuple(
            (size if axis in axes else 1) for axis, size in enumerate(self.shape)
        )
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
        amount = prod(window_sizes)
        px = self.encode_as(encoding(type(self)))
        dself = px.discretization
        dr = Discretization(
            (dself.domain.lo * amount, dself.domain.hi * amount),
            (dself.codomain.lo * amount, dself.codomain.hi * amount),
            dself.flip,
        )
        result = px._rolling_sum_(window_sizes, dr)
        assert result.shape == tuple(
            (s - w + 1) for s, w in zip(self.shape, window_sizes, strict=False)
        )
        assert result.discretization == dr
        return result

    def _rolling_sum_(self, window_sizes: tuple[int, ...], dr: Discretization) -> Self:
        _, _ = window_sizes, dr
        raise MissingMethod(self, "computing the sum of")

    # average

    def average(self, axis: int | tuple[int, ...] = 0, keepdims: bool = False) -> Pixels:
        """
        The average of all values along the specified axis or axes.
        """
        rank = self.rank
        axes = canonicalize_axes(axis, rank)
        window_sizes = tuple(
            (size if axis in axes else 1) for axis, size in enumerate(self.shape)
        )
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

    def median(self, axis: int | tuple[int, ...] = 0, keepdims: bool = False) -> Pixels:
        """
        The median of all values along the specified axis or axes.
        """
        rank = self.rank
        axes = canonicalize_axes(axis, rank)
        window_sizes = tuple(
            (size if axis in axes else 1) for axis, size in enumerate(self.shape)
        )
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
        assert result.shape == tuple(
            (s - w + 1) for s, w in zip(self.shape, window_sizes, strict=False)
        )
        return result

    def _rolling_median_(self, window_sizes: tuple[int, ...]) -> Self:
        _ = window_sizes
        raise MissingMethod(self, "computing the median of")

    # TODO new methods: variance, convolve, fft


P = TypeVar("P", bound=Pixels)


@dataclass(frozen=True)
class PixelsInitializer(Initializer[P]):
    pass


###############################################################################
###
### Auxiliary Functions


class MissingMethod(TypeError):
    instance: object
    action: str

    def __init__(self, instance: object, action: str):
        super().__init__(f"No method for {action} objects of type {type(instance)}.")
        self.instance = instance
        self.action = action


class MissingClassmethod(TypeError):
    cls: type
    action: str

    def __init__(self, cls: type, action: str):
        super().__init__(f"No method for {action} objects of type {cls}.")
        self.cls = cls
        self.action = action


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
    ituple: tuple
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
    for entry, size in zip(ituple, shape, strict=False):
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


def canonicalize_axes(axis: int | tuple[int, ...], rank: int) -> tuple[int, ...]:
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
    window_size: int | tuple[int, ...], shape: tuple[int, ...]
) -> tuple[int, ...]:
    rank = len(shape)
    if isinstance(window_size, int):
        window_sizes = (window_size,) + (1,) * max(0, rank - 1)
    elif isinstance(window_size, tuple):
        n = len(window_size)
        window_sizes = window_size + (1,) * max(0, rank - n)
    else:
        raise TypeError(f"Invalid window size specifier {window_size}.")
    if len(window_sizes) > rank:
        raise ValueError(f"Too many window sizes for data of rank {rank}.")
    for ws, size in zip(window_sizes, shape, strict=False):
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
    pxa, pxb = pixelize(a, b)
    shape = broadcast_shapes(pxa.shape, pxb.shape)
    return pxa.broadcast_to(shape), pxb.broadcast_to(shape)


def dtype_black_white_states(dtype: npt.DTypeLike) -> tuple[Real, Real, int]:
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return (0.0, 1.0, 2**53)
    if dtype == np.float32:
        return (0.0, 1.0, 2**24)
    elif dtype.kind == "u":
        nbits = dtype.itemsize * 8
        return (0, 2**nbits - 1, 2**nbits)
    elif dtype.kind == "i":
        nbits = dtype.itemsize * 8
        return (-(2 ** (nbits - 1)), (2 ** (nbits - 1)) - 1, 2**nbits)
    else:
        raise TypeError(f"Cannot convert {dtype} objects to pixels values.")
