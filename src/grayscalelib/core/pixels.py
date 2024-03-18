from __future__ import annotations

from abc import abstractmethod

from contextlib import contextmanager

from math import ceil, log2, prod

from types import EllipsisType

from typing import Callable, Iterable, Self, Sequence, TypeVar

from encodable import choose_encoding, encode_as, Encodable

from grayscalelib.core.protocols import ArrayLike, RealLike

from grayscalelib.core.simplearray import SimpleArray


_mandatory_pixels_type: type[Pixels] | None = None

_default_pixels_type: type[Pixels] | None = None

_default_power: int = -12


def set_default_pixels_type(cls: type[Pixels]):
    global _default_pixels_type
    _default_pixels_type = cls


def encoding(cls, *clss) -> type[Pixels]:
    if _mandatory_pixels_type:
        return _mandatory_pixels_type
    elif _default_pixels_type is None:
        return choose_encoding(cls, *clss)
    else:
        return choose_encoding(cls, *clss, _default_pixels_type)


@contextmanager
def pixels_type(pt: type[Pixels], /):
    """
    Create a context in which all operations on pixels will be carried out
    using the supplied representation.
    """
    global _mandatory_pixels_type
    previous = _mandatory_pixels_type
    _mandatory_pixels_type = pt
    try:
        yield pt
    finally:
        _mandatory_pixels_type = previous


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
def default_power(power: int):
    """
    Create a context in which the supplied power is the default when
    constructing pixels.
    """
    global _default_power
    previous = _default_power
    _default_power = power
    try:
        yield power
    finally:
        _default_power = previous


class Pixels(Encodable):
    """A container for non-negative values with uniform spacing.

    This class describes an abstract protocol for working with grayscale data.
    It supports working with individual values, vectors of values, images of
    values, videos of values, and stacks thereof.  Each pixel value is encoded
    in the form datum*(2**power), where datum is an integer that may be
    different for each pixel, and power is an integer that is the same for all
    pixels.
    """
    def __new__(cls, data, **kwargs):
        # If someone attempts to instantiate the abstract pixels base class,
        # instantiate an appropriate subclass instead.
        if cls is Pixels:
            newcls = encoding(cls)
            assert newcls != cls
            return encoding(newcls).__new__(newcls, data, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(
            self,
            data: RealLike | Sequence | ArrayLike,
            *,
            black: RealLike = 0,
            white: RealLike = 1,
            limit: RealLike | None = None,
            power: int | None = None):
        """
        Initialize a Pixels container, based on the supplied arguments.

        Parameters
        ----------
        data: RealLike or Sequence or ArrayLike
            A real number, a nested sequence of real numbers, or an object that
            has a shape and that returns real numbers when indexed.  Real numbers
            are treated as arrays of rank zero.  Nested sequences of real numbers
            must be structured such that all sequences of the same depth have the
            same length, and they are treated as arrays whose shape consists
            of these lengths.
        black: real
            The number that is mapped to the intensity zero when converting data
            to pixels.  All data smaller than or equal to this number is treated
            as pixels with intensity zero.
        white: real
            The number that is mapped to the intensity one when converting data
            to pixels.
        limit: real
            The number at which input data saturates.  Defaults to white, but can
            be set to a higher value to allow for pixel values larger than one.
        power: int, optional
            The exponent in the datum*(2**power) encoding of each pixel value.
            Most pixel operations clip their result to the [0, 1] interval, so
            power values are usually be negative.  A power of -1 means pixel
            values can be represented with a precision of 0.5, and a power of -3
            means that values can be represented with a precision of 0.125.
        """
        if power is None:
            power = _default_power
        if limit is None:
            limit = white
        if not (black < white):
            raise ValueError("Black must be less than white.")
        if not (black <= limit):
            raise ValueError("Black must be less than or equal to limit.")
        array = coerce_to_array(data)
        self._init_(array, black, white, limit, power)
        # Ensure the container was initialized correctly.
        assert self.shape == array.shape
        assert self.power == power
        assert self.limit == round((limit - black) / ((white - black) * (2**power)))

    @classmethod
    def from_data(cls, data: ArrayLike, power: int, limit: int):
        return cls(data, white=2**(-power), power=power, limit=limit)

    @abstractmethod
    def _init_(
            self,
            array: ArrayLike,
            black: RealLike,
            white: RealLike,
            limit: RealLike,
            power: int):
        """
        Initialize the supplied pixels based on the supplied parameters.  This
        method is called by __init__ after ensuring that all supplied arguments
        are well formed.
        """
        raise MissingMethod(self, "creating")

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        A tuple that describes the size of each axis.
        """
        ...

    @property
    @abstractmethod
    def power(self) -> int:
        """
        The power of two that is multiplied with each internal datum to produce
        a pixel's value.
        """
        ...

    @property
    def scale(self) -> float:
        """
        The minimum possible interval between any two differing pixel values.
        """
        return 2.0 ** self.power

    @property
    @abstractmethod
    def data(self) -> ArrayLike:
        """
        A read-only array of integers of the same shape as the pixels, that
        contains the pixel values before they have been multiplied with
        2**power.
        """
        ...

    @property
    def black(self) -> int:
        """
        The smallest integer that may appear in any pixel datum.
        """
        return 0

    @property
    def white(self) -> int:
        """
        The pixel datum that corresponds to a value of one.

        Raises an error if the power of the container is positive, because such
        an encoding cannot describe a value of one precisely anymore.
        """
        if self.power <= 0:
            return 2**(-self.power)
        else:
            raise RuntimeError("Pixels with positive power have no exact white.")

    @property
    @abstractmethod
    def limit(self) -> int:
        """
        The largest integer that may appear in any pixel datum.
        """
        ...

    @property
    def rank(self) -> int:
        """
        The number of axes of this container.
        """
        return len(self.shape)

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
        data = coerce_to_nested_sequence(self.data)
        return f"{name}({data}, shape={self.shape}, power={self.power}, limit={self.limit})"

    @abstractmethod
    def to_array(self) -> ArrayLike:
        raise MissingMethod(self, "reifying")

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
        cls = encoding(type(self))
        # Call the actual implementation.
        result = encode_as(self, cls)._permute_(permutation)
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

    # reshape

    def reshape(self, shape: tuple[int, ...]) -> Pixels:
        """
        Returns pixels with the original data and the supplied shape.
        """
        if prod(shape) != prod(self.shape):
            raise ValueError(f"Cannot reshape from shape {self.shape} to shape {shape}.")
        cls = encoding(type(self))
        result = encode_as(self, cls)._reshape_(shape)
        assert result.shape == shape
        assert result.power == self.power
        assert result.limit == self.limit
        return result

    def _reshape_(self, shape: tuple[int, ...]) -> Self:
        _ = shape
        raise MissingMethod(self, "reshaping")

    # broadcast_to

    def broadcast_to(self, shape: tuple[int, ...]) -> Pixels:
        """
        Replicate and stack the supplied data until it has the specified shape.
        """
        result = self._broadcast_to_(shape)
        assert result.shape == shape
        assert result.power == self.power
        assert result.limit == self.limit
        return result

    def _broadcast_to_(self, shape: tuple[int, ...]) -> Pixels:
        _ = shape
        raise MissingMethod(self, "broadcasting")

    # predicates

    def __bool__(self) -> bool:
        raise RuntimeError("Never boolify Pixels, use .any() or .all() instead.")

    def any(self) -> bool:
        """
        Whether at least one pixel in the container has a non-zero value.
        """
        cls = encoding(type(self))
        result = encode_as(self, cls)._any_()
        assert result is True or result is False
        return result

    def _any_(self) -> bool:
        raise MissingMethod(self, "testing for any-non-zero")

    def all(self) -> bool:
        """
        Whether all pixels in the container have a non-zero value.
        """
        cls = encoding(type(self))
        result = encode_as(self, cls)._all_()
        assert result is True or result is False
        return result

    def _all_(self) -> bool:
        raise MissingMethod(self, "testing for all-non-zero")


    # and

    def __and__(self, other) -> Pixels:
        """
        The logical conjunction of the two supplied containers.

        The resulting container has a power of zero and a limit of zero or one.
        """
        a, b = broadcast(self, other)
        result = a._and_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    __rand__ = __and__  # and is symmetric

    def _and_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "computing the logical conjunction of")

    # or

    def __or__(self, other) -> Pixels:
        """
        The logical disjunction of the two supplied containers.

        The resulting container has a power of zero and a limit of zero or one.
        """
        a, b = broadcast(self, other)
        result = a._or_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    __ror__ = __or__ # or is symmetric

    def _or_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "computing the logical disjunction of")

    # xor

    def __xor__(self, other) -> Pixels:
        """
        The exclusive disjunction of the two supplied containers.

        The resulting container has a power of zero and a limit of zero or one.
        """
        a, b = broadcast(self, other)
        result = a._xor_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    __rxor__ = __or__ # xor is symmetric

    def _xor_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "logical xor-ing")

    # lshift

    def __lshift__(self, amount: int) -> Pixels:
        """
        Multiply each value by two to the power of the supplied amount.

        The power of the result is the power of the supplied container plus the
        supplied amount.
        """
        # Ensure the amount is non-negative.
        if amount < 0:
            return self >> -amount
        cls = encoding(type(self))
        pix = encode_as(self, cls)
        # Handle the trivial case where the amount is zero.
        if amount == 0:
            return pix
        result = pix._lshift_(amount)
        assert result.shape == self.shape
        assert result.power == self.power + amount
        assert result.limit == self.limit
        return result

    def _lshift_(self: Self, amount: int) -> Self:
        _ = amount
        raise MissingMethod(self, "increasing the scale of")

    # rshift

    def __rshift__(self, amount: int) -> Pixels:
        """
        Divide each value by two to the power of the supplied amount.

        The power of the result is the power of the supplied container minus
        the supplied amount.
        """
        # Ensure the amount is non-negative.
        if amount < 0:
            return self << -amount
        cls = encoding(type(self))
        pix = encode_as(self, cls)
        if amount == 0:
            return pix
        result = pix._rshift_(amount)
        assert result.shape == self.shape
        assert result.power == self.power - amount
        assert result.limit == self.limit
        return result

    def _rshift_(self: Self, amount: int) -> Self:
        _ = amount
        raise MissingMethod(self, "decreasing the scale of")

    # abs

    def __abs__(self) -> Pixels:
        """
        Do nothing, since pixels are non-negative by definition.
        """
        cls = encoding(type(self))
        result = encode_as(self, cls)
        assert result.shape == self.shape
        assert result.power == self.power
        assert result.limit == self.limit
        return result

    # invert

    def __invert__(self) -> Pixels:
        """
        One minus the original value, clipped to [0, 1]

        The resulting container has the same power of P = min(power, 0), and a
        limit of 2**abs(P).
        """
        cls = encoding(type(self))
        result = encode_as(self, cls)._invert_()
        power = min(self.power, 0)
        limit = 2**abs(power)
        assert result.shape == self.shape
        assert result.power == power
        assert result.limit == limit
        return result

    def _invert_(self) -> Self:
        raise MissingMethod(self, "inverting")

    # neg

    def __neg__(self) -> Pixels:
        """
        Return zeros of the same shape.

        This is a weird operator for pixels.  The convention is that the result
        of each pixel math operation is clipped to the [0, 1] interval.  For
        negation, this means that the resulting array is all zero.
        """
        return type(self)(0, power=self.power, limit=0).broadcast_to(self.shape)

    # pos

    def __pos__(self) -> Pixels:
        """
        Do nothing, since pixels are non-negative by definition.
        """
        cls = encoding(type(self))
        result = encode_as(self, cls)
        assert result.shape == self.shape
        assert result.limit == self.limit
        assert result.power == self.power
        return result

    # add

    def __add__(self, other):
        """
        Add the values of the two containers and clip the result to [0, 1].

        The result of the addition of two containers A and B has a power of P =
        min(A.power, B.power, 0), and a limit that is the minimum of
        round(A.limit * 2**(P - A.power) + B.limit * 2**(P - B.power)) and
        2**abs(P).
        """
        a, b = broadcast(self, other)
        result = a._add_(b)
        power = min(a.power, b.power, 0)
        limit = min(round(a.limit * 2**(power - a.power) +
                          b.limit * 2**(power - b.power)),
                    2**abs(power))
        assert result.shape == a.shape
        assert result.power == power
        assert result.limit == limit
        return result

    __radd__ = __add__# add is symmetric

    def _add_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "adding")

    # sub

    def __sub__(self, other):
        """
        Subtract the values of the two containers and clip the result to the
        interval [0, 1].

        The result of the subtraction of two containers A and B has a power of
        P = min(A.power, B.power, 0), and a limit that is the minimum of
        round(A.limit * 2**(P - a.power)) and 2**abs(P).
        """
        a, b = broadcast(self, other)
        result = a._sub_(b)
        power = min(a.power, b.power, 0)
        limit = min(round(a.limit * 2**(a.power - power)), 2**abs(power))
        assert result.shape == a.shape
        assert result.power == power
        assert result.limit == limit
        return result

    def __rsub__(self, other):
        b, a = broadcast(self, other)
        result = a._sub_(b)
        power = min(a.power, b.power)
        limit = a.limit * 2**(power - a.power)
        assert result.shape == a.shape
        assert result.power == power
        assert result.limit == limit
        return result

    def _sub_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "subtracting")

    # mul

    def __mul__(self, other) -> Pixels:
        """
        Multiply the values of the containers and clip the result to [0, 1].

        The result of the multiplication of some containers A and B has the
        power of P = min(A.power, B.power, 0), and a limit of 2**abs(P)
        """
        a, b = broadcast(self, other)
        result = a._mul_(b)
        power = min(a.power, b.power, 0)
        limit = 2**abs(power)
        assert result.shape == a.shape
        assert result.power == power
        assert result.limit == limit
        return result

    __rmul__ = __mul__ # mul is symmetric

    def _mul_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "multiplying")

    # pow

    def __pow__(self, exponent: RealLike) -> Pixels:
        """
        Raise each value to the specified power, and clip the result to [0, 1].

        The result of A**B has a power P = min(A.power, 0), and a limit of
        min(2**abs(power), round((A.limit ** B) * 2**(B * A.power - power)))
        """
        a = encode_as(self, encoding(type(self)))
        b = float(exponent)
        power = min(a.power, 0)
        limit = min(2**abs(power), round((a.limit ** b) * 2**(b * a.power - power)))
        result = a._pow_(b)
        assert result.shape == self.shape
        assert result.power == power
        assert result.limit == limit
        return result

    def _pow_(self: Self, other: float) -> Self:
        _ = other
        raise MissingMethod(self, "exponentiating")

    # truediv

    def __truediv__(self, other) -> Pixels:
        """
        Divide the values of the two containers and clip the result to [0, 1].

        The power P of A/B is min(0, A.power - B.power - ceil(log2(B.limit))),
        and the corresponding limit is 2**abs(P).
        """
        a, b = broadcast(self, other)
        power = min(0, a.power - b.power - ceil(log2(b.limit)))
        result = a._truediv_(b)
        assert result.shape == a.shape
        assert result.power == power
        assert result.limit == 2**abs(power)
        return result

    def _truediv_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "dividing")

    # floordiv

    def __floordiv__(self, other) -> Pixels:
        """
        Zero wherever the division is less than one, one otherwise.

        The result has a power of zero and a limit of at most one.
        """
        a, b = broadcast(self, other)
        result = a._floordiv_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    def __rfloordiv__(self, other) -> Pixels:
        b, a = broadcast(self, other)
        result = a._floordiv_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    def _floordiv_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "dividing")

    # mod

    def __mod__(self, other) -> Pixels:
        """
        Left value modulo right value, clipped to [0, 1].

        For any containers A and B, we have (A // B) + (A % B) == A.clip(0, 1).
        """
        a, b = broadcast(self, other)
        result = a._mod_(b)
        power = min(a.power, b.power, 0)
        limit = 2**abs(power)
        assert result.shape == a.shape
        assert result.power == power
        assert result.limit == limit
        return result

    def __rmod__(self, other) -> Pixels:
        b, a = broadcast(self, other)
        result = a._mod_(b)
        power = min(a.power, b.power, 0)
        limit = 2**abs(power)
        assert result.shape == a.shape
        assert result.power == power
        assert result.limit == limit
        return result

    def _mod_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "computing the modulus of")

    # lt

    def __lt__(self, other) -> Pixels:
        """
        One wherever the left value is smaller than the right, zero otherwise.

        The resulting container has one integer bit, and zero fractional bits.
        """
        a, b = broadcast(self, other)
        result = a._lt_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    def _lt_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "comparing")

    # gt

    def __gt__(self, other) -> Pixels:
        """
        One wherever the left value is greater than the right, zero otherwise.

        The resulting container has one integer bit, and zero fractional bits.
        """
        a, b = broadcast(self, other)
        result = a._gt_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    def _gt_(self: Self, other: Self) -> Self:
        return other._lt_(self)

    # le

    def __le__(self, other) -> Pixels:
        """
        One wherever the left value is less than or equal to the right, zero
        otherwise.

        The resulting container has one integer bit, and zero fractional bits.
        """
        a, b = broadcast(self, other)
        result = a._le_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    def _le_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "comparing")

    # ge

    def __ge__(self, other) -> Pixels:
        """
        One wherever the left value is greater than or equal to the right, zero
        otherwise.

        The resulting container has one integer bit, and zero fractional bits.
        """
        a, b = broadcast(self, other)
        result = a._ge_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    def _ge_(self: Self, other: Self) -> Self:
        return other._le_(self)

    # eq

    def __eq__(self, other) -> Pixels: # type: ignore
        """
        One wherever the left value is equal to the right, zero otherwise.

        The resulting container has one integer bit, and zero fractional bits.
        """
        a, b = broadcast(self, other)
        result = a._eq_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
        return result

    def _eq_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "determining the equality of")

    # ne

    def __ne__(self, other) -> Pixels: # type: ignore
        """
        One wherever the left value is different than the right, zero otherwise.

        The resulting container has one integer bit, and zero fractional bits.
        """
        a, b = broadcast(self, other)
        result = a._ne_(b)
        assert result.shape == a.shape
        assert result.power == 0
        assert result.limit <= 1
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
        result = encode_as(self, cls)._rolling_sum_(window_sizes)
        assert result.shape == tuple((s - w + 1) for s, w in zip(self.shape, window_sizes))
        assert result.limit == prod(window_sizes) * self.limit
        assert result.power == self.power
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
        result = encode_as(self, cls)._rolling_median_(window_sizes)
        assert result.shape == tuple((s - w + 1) for s, w in zip(self.shape, window_sizes))
        assert result.limit == self.limit
        assert result.power == self.power
        return result

    def _rolling_median_(self, window_sizes: tuple[int, ...]) -> Self:
        _ = window_sizes
        raise MissingMethod(self, "computing the median of")

    # TODO new methods: variance, convolve, fft


def MissingMethod(self, action) -> TypeError:
    return TypeError(f"No method for {action} objects of type {type(self)}.")

def MissingClassmethod(cls, action) -> TypeError:
    return TypeError(f"No classmethod for {action} of type {cls}.")


def coerce_to_array(data) -> ArrayLike:
    # Case 1 - Data is already an array.
    if isinstance(data, ArrayLike):
        return data
    # Case 2 - Data is a nested sequence.
    elif isinstance(data, Sequence):
        sizes = []
        rest = data
        while isinstance(rest, Sequence):
            size = len(rest)
            sizes.append(size)
            if size == 0:
                break
            rest = rest[0]
        shape = tuple(sizes)
        # Check whether the shape fits the data, and copy the contents of all
        # innermost sequences to the resulting values.
        values = []
        rank = len(shape)
        def check_and_copy(seq, depth):
            if len(seq) != shape[depth]:
                raise ValueError(
                    f"List of length {len(seq)} in axis of size {shape[depth]}."
                )
            if depth == rank - 1:
                values.extend(seq)
                return
            for elt in seq:
                check_and_copy(elt, depth+1)
        check_and_copy(data, 0)
        return SimpleArray(values, shape)
    # Case 3 - Data is a scalar.
    elif isinstance(data, RealLike):
        return SimpleArray([data], ())
    else:
        raise TypeError(f"Cannot coerce {data} to an array.")


Seq = TypeVar("Seq")

def coerce_to_nested_sequence(
        data,
        constructor: Callable[[Iterable], Seq] = list) -> Seq | RealLike:
    if isinstance(data, ArrayLike):
        def sequify(index, shape) -> Seq | RealLike:
            if shape == ():
                return data[*index]
            return constructor(sequify(index + (i,), shape[1:]) for i in range(shape[0]))
        return sequify((), data.shape)
    elif isinstance(data, RealLike):
        return data
    elif isinstance(data, Sequence):
        def convert(elt):
            if not isinstance(elt, Sequence):
                assert isinstance(elt, RealLike)
                return elt
            else:
                return constructor(convert(x) for x in elt)
        return convert(data)
    else:
        raise TypeError(f"Cannot coerce {data} to a nested sequence.")


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


def broadcast_shapes(shape1: tuple, shape2: tuple) -> tuple:
    """Broadcast the two supplied shapes or raise an error."""
    rank1 = len(shape1)
    rank2 = len(shape2)
    for axis in range(min(rank1, rank2)):
        if shape1[axis] != shape2[axis]:
            raise ValueError(f"Size mismatch in axis {axis}.")
    if rank1 < rank2:
        return shape2
    else:
        return shape1


def broadcast(a, b) -> tuple[Pixels, Pixels]:
    """
    Ensure the two supplied containers have the same shape and class.

    If either argument is not a pixels container but a real number, convert it
    to a suitable container with the same power as the other one.  If both
    arguments are real numbers, raise an error.
    """
    if isinstance(a, Pixels):
        if isinstance(b, Pixels):
            cls = encoding(type(a), type(b))
            pxa = encode_as(a, cls)
            pxb = encode_as(b, cls)
            shape = broadcast_shapes(pxa.shape, pxb.shape)
            return (pxa.broadcast_to(shape), pxb.broadcast_to(shape))
        else:
            cls = encoding(type(a))
            pxa = encode_as(a, cls)
            pxb = cls(b, power=pxa.power, limit=b)
            return (pxa, pxb.broadcast_to(pxa.shape))
    else:
        if isinstance(b, Pixels):
            cls = encoding(type(b))
            pxb = encode_as(b, cls)
            pxa = cls(a, power=pxb.power, limit=a)
            return (pxa.broadcast_to(pxb.shape), pxb)
        else:
            raise TypeError("Cannot broadcast two scalars.")
