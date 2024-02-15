from __future__ import annotations

from abc import abstractmethod

from contextlib import contextmanager

from fractions import Fraction

from math import floor

from types import EllipsisType

from typing import Callable, Iterable, Self, Sequence, TypeVar

from encodable import choose_encoding, encode_as, Encodable

from grayscalelib.core.protocols import ArrayLike, RealLike

from grayscalelib.core.simplearray import SimpleArray


_mandatory_pixels_type: type[Pixels] | None = None

_default_pixels_type: type[Pixels] | None = None

_default_fbits: int = 12


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
def default_fbits(fbits: int):
    """
    Create a context in which the supplied number of fractional bits
    is the default when constructing pixels.
    """
    global _default_fbits
    previous = _default_fbits
    _default_fbits = fbits
    try:
        yield fbits
    finally:
        _default_fbits = previous


class Pixels(Encodable):
    """A container for non-negative values with uniform spacing.

    This class describes an abstract protocol for working with grayscale data.
    It supports working with individual values, vectors of values, images of
    values, videos of values, and stacks thereof.  Each individual value is
    represented using a fixed-point fractional encoding in base two, with some
    number of bits describing the integer part of the encoding, and another
    number of bits describing the fractional part.  The number of integer bits
    and fractional bits must be the same for all values in the container.
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
            ibits: int | None = None,
            fbits: int | None = None):
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
        ibits: int, optional
            The number of bits of integer precision of the resulting pixels.
        fbits: int, optional
            The number of bits of fractional precision of the resulting pixels.
            Defaults to the maximum fractional precision of any item in the data
            parameter.
        """
        if limit is None:
            limit = white
        if not (black <= white):
            raise ValueError("Black must be less than or equal to white.")
        if not (black <= limit):
            raise ValueError("Black must be less than or equal to limit.")
        delta = (white - black)
        if ibits is None:
            ibits = 0 if delta == 0 else floor((limit - black) / delta).bit_length()
        if ibits < 0:
            raise ValueError("The number of integer bits must be non-negative.")
        if fbits is None:
            fbits = _default_fbits
        if fbits < 0:
            raise ValueError("The number of fractional bits must be non-negative.")
        scale = 0 if delta == 0 else divide_accurately((1 << fbits), delta)
        offset = -1 * black * scale
        maxval = min(round(limit * scale + offset), (1 << (ibits + fbits)) - 1)
        array = coerce_to_array(data)
        self._init_(array, ibits, fbits, scale, offset, maxval)
        # Ensure the container was initialized correctly.
        assert self.ibits == ibits
        assert self.fbits == fbits
        assert self.shape == array.shape

    @classmethod
    def from_numerators(cls, array: ArrayLike, ibits: int, fbits: int):
        white = (1 << fbits)
        limit = (white << ibits) - 1
        return cls(array, white=white, limit=limit, ibits=ibits, fbits=fbits)

    @abstractmethod
    def _init_(
            self,
            array: ArrayLike,
            ibits: int,
            fbits: int,
            scale: RealLike,
            offset: RealLike,
            maxval: int):
        """
        Initialize the supplied pixels based on the supplied parameters.

        Parameters
        ----------

        data: ArrayLike
            An array-like object of real numbers.  Each element of the array
            provides the data for one pixel.  The conversion of numbers to
            pixels is governed by the remaining arguments.
        ibits: int
            The number of bits of integer precision of the resulting pixels.
        fbits: int
            The number of bits of fractional precision of the resulting pixels.
        scale, offset, maxval: RealLike
            These numbers govern the mapping from data elements to integers that
            are the numerators of a fractional encoding with denominator 2**fbits.
            The formula for converting the number X to the corresponding numerator
            is max(0, min(round(scale * number[I] + offset), maxval)).
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
    def ibits(self) -> int:
        """
        The number of bits of precision of the integer part of each pixel
        value.
        """
        ...

    @property
    @abstractmethod
    def fbits(self) -> int:
        """
        The number of bits of precision of the fractional part of each pixel
        value.
        """
        ...

    @property
    def denominator(self) -> int:
        """
        The denominator of the internal fractional encoding of each pixel
        value.
        """
        return 2 ** self.fbits

    @property
    @abstractmethod
    def numerators(self) -> ArrayLike:
        """
        A read-only array of integers of the same shape as the pixels, whose
        values are the numerators of each pixel value.
        """
        pass

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
        A textual representation for this container.
        """
        name = type(self).__name__
        data = coerce_to_nested_sequence(self.numerators)
        return f"{name}({data}, white={self.denominator}, ibits={self.ibits}, fbits={self.fbits})"

    @abstractmethod
    def to_array(self) -> ArrayLike:
        raise MissingMethod(self, "reifying")

    # getitem

    def __getitem__(self: Pixels, index: EllipsisType | int | slice | tuple[int | slice, ...]) -> Pixels:
        """Select a particular index or slice of one or  more axes."""
        return self._getitem_(canonicalize_index(index, self.shape))

    def _getitem_(self, index: tuple[int | slice, ...]) -> Pixels:
        _ = index
        raise MissingMethod(self, "indexing")

    # permute

    def permute(self, p0: int | tuple = (), /, *more: int) -> Pixels:
        """Reorder all axes according to the supplied integers."""
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

    # broadcast_to

    def broadcast_to(self, shape: tuple[int, ...]) -> Pixels:
        """
        Replicate and stack the supplied data until it has the specified shape.
        """
        result = self._broadcast_to_(shape)
        assert result.shape == shape
        assert result.ibits == self.ibits
        assert result.fbits == self.fbits
        return result

    def _broadcast_to_(self, shape: tuple[int, ...]) -> Pixels:
        _ = shape
        raise MissingMethod(self, "broadcasting")

    # bool

    def __bool__(self) -> bool:
        """Whether at least one pixel in the container is not zero."""
        cls = encoding(type(self))
        result = encode_as(self, cls)._bool_()
        assert result is True or result is False
        return result

    def _bool_(self) -> bool:
        raise MissingMethod(self, "determining the truth of")

    # not

    def __not__(self) -> Pixels:
        """
        One wherever the supplied container is zero, and zero otherwise.

        The resulting container has one integer bit, and zero fractional bits.
        """
        cls = encoding(type(self))
        result = encode_as(self, cls)._not_()
        assert result.shape == self.shape
        assert result.fbits == 0
        assert result.ibits == 1
        return result

    def _not_(self: Self) -> Self:
        raise MissingMethod(self, "logically negating")

    # and

    def __and__(self, other) -> Pixels:
        """
        The logical conjunction of the two supplied containers.

        The resulting container has one integer bit, and zero fractional bits.
        """
        a, b = broadcast(self, other)
        result = a._and_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == 0
        return result

    __rand__ = __and__  # and is symmetric

    def _and_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "computing the logical conjunction of")

    # or

    def __or__(self, other) -> Pixels:
        """
        The logical disjunction of the two supplied containers.

        The resulting container has one integer bit, and zero fractional bits.
        """
        a, b = broadcast(self, other)
        result = a._or_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == 0
        return result

    __ror__ = __or__ # or is symmetric

    def _or_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "computing the logical disjunction of")

    # xor

    def __xor__(self, other) -> Pixels:
        """The exclusive disjunction of the two supplied containers."""
        a, b = broadcast(self, other)
        result = a._xor_(b)
        assert result.shape == a.shape
        return result

    __rxor__ = __or__ # xor is symmetric

    def _xor_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "logical xor-ing")

    # lshift

    def __lshift__(self, amount: int) -> Pixels:
        """
        Multiply each value by two to the power of the supplied amount.

        While shifting, move the specified amount of bits from the fractional
        part of the encoding to the integer part of the encoding.
        """
        # Ensure the amount is non-negative.
        if amount < 0:
            return self >> -amount
        cls = encoding(type(self))
        pix = encode_as(self, cls)
        if amount == 0:
            return pix
        result = pix._lshift_(amount)
        assert result.shape == self.shape
        assert max(0, self.ibits + amount) == result.ibits
        assert max(0, self.fbits - amount) == result.fbits
        return result

    def _lshift_(self: Self, amount: int) -> Self:
        _ = amount
        raise MissingMethod(self, "increasing the precision of")

    # rshift

    def __rshift__(self, amount: int) -> Pixels:
        """
        Divide each value by two to the power of the supplied amount.

        While shifting, move the specified amount of bits from the integer
        part of the encoding to the fractional part of the encoding.
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
        assert max(0, self.ibits - amount) == result.ibits
        assert max(0, self.fbits + amount) == result.fbits
        return result

    def _rshift_(self: Self, amount: int) -> Self:
        _ = amount
        raise MissingMethod(self, "decreasing the precision of")

    # abs

    def __abs__(self) -> Pixels:
        """
        Do nothing, since pixels are never negative.
        """
        cls = encoding(type(self))
        result = encode_as(self, cls)._abs_()
        assert result.shape == self.shape
        assert result.ibits == self.ibits
        assert result.fbits == self.fbits
        return result

    def _abs_(self: Self) -> Self:
        # Pixels are non-negative by definition
        return self

    # invert

    def __invert__(self) -> Pixels:
        """
        One minus the original value.

        The resulting container has one integer bit, and the same amount of
        fractional bits as the original one.
        """
        cls = encoding(type(self))
        result = encode_as(self, cls)._invert_()
        assert result.shape == self.shape
        assert result.ibits == 1
        assert result.fbits == self.fbits
        return result

    def _invert_(self: Self) -> Self:
        raise MissingMethod(self, "inverting")

    # neg

    # This is a weird operator for pixels.  The rule is that each pixel math
    # operation is carried out just as expected, but the result is clipped to
    # the [0, 1] interval.  For negation, this means that the resulting array
    # is all zero.
    def __neg__(self) -> Pixels:
        """
        Return zeros of the same shape.
        """
        return type(self)(0, ibits=0, fbits=self.fbits).broadcast_to(self.shape)

    # pos

    def __pos__(self) -> Pixels:
        """
        Do nothing, since pixels are non-negative by definition.
        """
        cls = encoding(type(self))
        result = encode_as(self, cls)._pos_()
        assert result.shape == self.shape
        assert result.ibits == self.ibits
        assert result.fbits == self.fbits
        return result

    def _pos_(self: Self) -> Self:
        # Pixels are non-negative by definition.
        return self

    # add

    def __add__(self, other):
        """
        Add the values of the two containers and clip the result to [0, 1].

        The result of this operation has exactly one integer bit, and a number
        of fractional bits that is the maximum of the numbers of fractional
        bits of the two arguments.
        """
        a, b = broadcast(self, other)
        result = a._add_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
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

        The result of this operation has exactly one integer bit, and a number
        of fractional bits that is the maximum of the numbers of fractional
        bits of the two arguments.
        """
        a, b = broadcast(self, other)
        result = a._sub_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
        return result

    def __rsub__(self, other):
        b, a = broadcast(self, other)
        result = a._sub_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
        return result

    def _sub_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "subtracting")

    # mul

    def __mul__(self, other) -> Pixels:
        """
        Add the values of the two containers and clip the result to [0, 1].

        The result of this operation has exactly one integer bit, and a number
        of fractional bits that is the maximum of the numbers of fractional
        bits of the two arguments.
        """
        a, b = broadcast(self, other)
        result = a._mul_(b)
        assert result.shape == a.shape
        return result

    __rmul__ = __mul__ # mul is symmetric

    def _mul_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'muling')

    # pow

    def __pow__(self, power) -> Pixels:
        """
        Raise each value to the specified power, and clip the result to [0, 1].

        The result of this operation has one integer bit, and a number of
        fractional bits that is the maximum of the numbers of fractional bits
        of the two arguments.
        """
        a, b = broadcast(self, power)
        result = a._pow_(b)
        assert result.shape == self.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
        return result

    def _pow_(self: Self, power: Self) -> Self:
        _ = power
        raise MissingMethod(self, "exponentiating")

    # truediv

    def __truediv__(self, other) -> Pixels:
        """
        Divide the values of the two containers and clip the result to [0, 1].

        The result of this operation has exactly one integer bit, and a number
        of fractional bits that is the maximum of the numbers of fractional
        bits of the two arguments.
        """
        a, b = broadcast(self, other)
        result = a._truediv_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
        return result

    def __rtruediv__(self, other) -> Pixels:
        b, a = broadcast(self, other)
        result = a._truediv_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
        return result

    def _truediv_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "dividing")

    # floordiv

    def __floordiv__(self, other) -> Pixels:
        """
        Divide the values of the two containers and clip the result to [0, 1].

        The result of this operation has exactly one integer bit, and a number
        of fractional bits that is the maximum of the numbers of fractional
        bits of the two arguments.
        """
        a, b = broadcast(self, other)
        result = a._floordiv_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
        return result

    def __rfloordiv__(self, other) -> Pixels:
        b, a = broadcast(self, other)
        result = a._floordiv_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
        return result

    def _floordiv_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "dividing")

    # mod

    def __mod__(self, other) -> Pixels:
        """
        Left value modulo right value, clipped to [0, 1].

        The result of this operation has exactly one integer bit, and a number
        of fractional bits that is the maximum of the numbers of fractional
        bits of the two arguments.
        """
        a, b = broadcast(self, other)
        result = a._mod_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
        return result

    def __rmod__(self, other) -> Pixels:
        b, a = broadcast(self, other)
        result = a._mod_(b)
        assert result.shape == a.shape
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
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
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
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
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
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
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
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
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
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
        assert result.ibits == 1
        assert result.fbits == max(a.fbits, b.fbits)
        return result

    def _eq_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, "determining the equality of")

    # TODO new methods: average, rolling_average, difference

    # TODO new methods: mean, median, variance, convolve, fft


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


def divide_accurately(a, b):
    if hasattr(a, 'as_integer_ratio') and hasattr(b, 'as_integer_ratio'):
        na, da = a.as_integer_ratio()
        nb, db = b.as_integer_ratio()
        frac = Fraction(na, da) / Fraction(nb, db)
        if frac.denominator == 1:
            return frac.numerator
        else:
            return frac
    else:
        # Use __truediv__ as a fallback solution.
        return a / b

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
    to a suitable container with the same fractional precision as the other
    one.  If both arguments are real numbers, raise an error.
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
            pxb = cls(b, fbits=pxa.fbits, limit=min(1, b))
            return (pxa, pxb.broadcast_to(pxa.shape))
    else:
        if isinstance(b, Pixels):
            cls = encoding(type(b))
            pxb = encode_as(b, cls)
            pxa = cls(a, fbits=pxb.fbits, limit=min(1, a))
            return (pxa.broadcast_to(pxb.shape), pxb)
        else:
            raise TypeError("Cannot broadcast two scalars.")
