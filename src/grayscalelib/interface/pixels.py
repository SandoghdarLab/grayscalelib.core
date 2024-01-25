from __future__ import annotations

from abc import abstractmethod

from types import EllipsisType

from typing import Self, Generic, TypeVar, TypeVarTuple

from encodable import choose_encoding, encode_as, Encodable

T = TypeVar('T')
Shape = TypeVarTuple('Shape')
Rest = TypeVarTuple('Rest')

Scalar = float

fallback_encoding: type | None = None


def encoding(cls, *clss):
    if fallback_encoding is None:
        return choose_encoding(cls, *clss)
    else:
        return choose_encoding(cls, *clss, fallback_encoding)


class Pixels(Encodable, Generic[*Shape]):

    @property
    @abstractmethod
    def shape(self) -> tuple[*Shape]: ...

    @property
    @abstractmethod
    def precision(self) -> int: ...

    @property
    def denominator(self) -> int:
        return 2 ** self.precision - 1

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __len__(self: Pixels[T, *tuple]) -> T:
        if len(self.shape) == 0:
            raise RuntimeError("A rank zero container has no length.")
        return self.shape[0]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} shape={self.shape} precision={self.precision}>"

    @classmethod
    def zeros(cls, shape: tuple[*Rest]) -> Pixels[*Rest]:
        result = cls._zeros_(shape)
        assert result.shape == shape
        assert result.precision == 0
        return result

    @classmethod
    def _zeros_(cls, shape: tuple[*Rest]) -> Pixels[*Rest]:
        _ = shape
        raise MissingClassmethod(cls, 'creating zeros')

    @classmethod
    def ones(cls, shape: tuple[*Rest]) -> Pixels[*Rest]:
        result = cls._ones_(shape)
        assert result.shape == shape
        assert result.precision == 1
        return result

    @classmethod
    def _ones_(cls, shape: tuple[*Rest]) -> Pixels[*Rest]:
        _ = shape
        raise MissingClassmethod(cls, 'creating ones')

    # getitem

    def __getitem__(self: Pixels, index: EllipsisType | int | slice | tuple[int | slice, ...]) -> Pixels:
        return self._getitem_(canonicalize_index(index, self.shape))

    def _getitem_(self, index: tuple[int | slice, ...]) -> Pixels[*tuple[int, ...]]:
        _ = index
        raise MissingMethod(self, 'indexing')

    # permute

    def permute(self, *permutation: int) -> Pixels:
        rank = self.rank
        if not isinstance(permutation, tuple):
            raise TypeError(f"A permutation must be a tuple, not a {type(permutation)}.")
        if not len(permutation) == rank:
            raise ValueError(f"Invalid permutation {permutation} for data of rank {rank}.")
        for i, p in enumerate(permutation):
            if not isinstance(p, int):
                raise TypeError(f"The permutation entry {p} is not an integer.")
            if not 0 <= p < rank:
                raise ValueError(f"Invalid permutation entry {p} for data of rank {rank}.")
            if p in permutation[:i]:
                raise ValueError(f"Duplicate entry {p} in permutation {permutation}.")
        cls = encoding(type(self))
        result = encode_as(self, cls)._permute_(permutation)
        oldshape = self.shape
        newshape = result.shape
        assert len(permutation) == len(oldshape) == len(newshape)
        for i, p in enumerate(permutation):
            assert newshape[i] == oldshape[p]
        return result

    def _permute_(self, permutation: tuple[int, ...]) -> Pixels:
        _ = permutation
        raise MissingMethod(self, 'permuting')

    # bool

    def __bool__(self) -> bool:
        cls = encoding(type(self))
        result = encode_as(self, cls)._bool_()
        assert result is True or result is False
        return result

    def _bool_(self) -> bool:
        raise MissingMethod(self, 'determining the truth of')

    # lshift

    def __lshift__(self: Pixels[*Shape], amount: int) -> Pixels[*Shape]:
        if amount < 0:
            return self.__rshift__(-amount)
        cls = encoding(type(self))
        pix = encode_as(self, cls)
        result = pix._lshift_(amount)
        assert result.shape == self.shape
        assert result.precision == (self.precision + amount)
        return result

    def _lshift_(self: Self, amount: int) -> Self:
        _ = amount
        raise MissingMethod(self, 'increasing the precision of')

    # rshift

    def __rshift__(self: Pixels[*Shape], amount: int) -> Pixels[*Shape]:
        if amount < 0:
            return self.__lshift__(-amount)
        cls = encoding(type(self))
        pix = encode_as(self, cls)
        result = pix._rshift_(amount)
        assert result.shape == self.shape
        assert result.precision == max(0, (self.precision - amount))
        return result

    def _rshift_(self: Self, amount: int) -> Self:
        _ = amount
        raise MissingMethod(self, 'decreasing the precision of')

    # pow

    def __pow__(self: Pixels[*Shape], exponent: float) -> Pixels[*Shape]:
        cls = encoding(type(self))
        result = encode_as(self, cls)._pow_(exponent)
        assert result.shape == self.shape
        return result

    def _pow_(self: Self, exponent: float) -> Self:
        _ = exponent
        raise MissingMethod(self, 'exponentiating')

    # abs

    def __abs__(self: Pixels[*Shape]) -> Pixels[*Shape]:
        cls = encoding(type(self))
        result = encode_as(self, cls)._abs_
        assert result.shape == self.shape
        return result

    def _abs_(self: Self) -> Self:
        raise MissingMethod(self, 'absing')

    # not

    def __not__(self: Pixels[*Shape]) -> Pixels[*Shape]:
        cls = encoding(type(self))
        result = encode_as(self, cls)._not_
        assert result.shape == self.shape
        return result

    def _not_(self: Self) -> Self:
        raise MissingMethod(self, 'noting')

    # invert

    def __invert__(self: Pixels[*Shape]) -> Pixels[*Shape]:
        cls = encoding(type(self))
        result = encode_as(self, cls)._invert_
        assert result.shape == self.shape
        return result

    def _invert_(self: Self) -> Self:
        raise MissingMethod(self, 'inverting')

    # neg

    def __neg__(self: Pixels[*Shape]) -> Pixels[*Shape]:
        cls = encoding(type(self))
        result = encode_as(self, cls)._neg_
        assert result.shape == self.shape
        return result

    def _neg_(self: Self) -> Self:
        raise MissingMethod(self, 'neging')

    # pos

    def __pos__(self: Pixels[*Shape]) -> Pixels[*Shape]:
        cls = encoding(type(self))
        result = encode_as(self, cls)._pos_
        assert result.shape == self.shape
        return result

    def _pos_(self: Self) -> Self:
        raise MissingMethod(self, 'posing')

    # broadcast_to

    def broadcast_to(self, shape: tuple[*Rest]) -> Pixels[*Rest]:
        result = self._broadcast_to_(shape)
        assert result.shape == shape
        return result

    def _broadcast_to_(self, shape: tuple[*Rest]) -> Pixels[*Rest]:
        _ = shape
        raise MissingMethod(self, 'broadcasting')

    # broadcast

    def __broadcast__(self: Pixels, other: Pixels | Scalar) -> tuple[Pixels, Pixels]:
        cls = encoding(type(self), type(other))
        a = encode_as(self, cls)
        b = encode_as(other, cls)
        s = broadcast_shapes(a.shape, b.shape)
        return (a.broadcast_to(s), b.broadcast_to(s))

    # add

    def __add__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._add_(b)
        assert result.shape == a.shape
        return result

    def __radd__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]:
        b, a = self.__broadcast__(other)
        return a._add_(b)

    def _add_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'adding')

    # and

    def __and__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._and_(b)
        assert result.shape == a.shape
        return result

    def __rand__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]:
        b, a = self.__broadcast__(other)
        return a._and_(b)

    def _and_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'anding')

    # floordiv

    def __floordiv__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._floordiv_(b)
        assert result.shape == a.shape
        return result

    def __rfloordiv__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]:
        b, a = self.__broadcast__(other)
        return a._floordiv_(b)

    def _floordiv_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'floordiving')

    # mod

    def __mod__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._mod_(b)
        assert result.shape == a.shape
        return result

    def __rmod__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]:
        b, a = self.__broadcast__(other)
        return a._mod_(b)

    def _mod_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'moding')

    # mul

    def __mul__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._mul_(b)
        assert result.shape == a.shape
        return result

    def __rmul__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]:
        b, a = self.__broadcast__(other)
        return a._mul_(b)

    def _mul_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'muling')

    # or

    def __or__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._or_(b)
        assert result.shape == a.shape
        return result

    def __ror__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]:
        b, a = self.__broadcast__(other)
        return a._or_(b)

    def _or_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'oring')

    # sub

    def __sub__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._sub_(b)
        assert result.shape == a.shape
        return result

    def __rsub__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]:
        b, a = self.__broadcast__(other)
        return a._sub_(b)

    def _sub_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'subing')

    # truediv

    def __truediv__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._truediv_(b)
        assert result.shape == a.shape
        return result

    def __rtruediv__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]:
        b, a = self.__broadcast__(other)
        return a._truediv_(b)

    def _truediv_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'truediving')

    # xor

    def __xor__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._xor_(b)
        assert result.shape == a.shape
        return result

    def __rxor__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]:
        b, a = self.__broadcast__(other)
        return a._xor_(b)

    def _xor_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'xoring')

    # lt

    def __lt__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._lt_(b)
        assert result.shape == a.shape
        return result

    def _lt_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'lting')

    # gt

    def __gt__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._gt_(b)
        assert result.shape == a.shape
        return result

    def _gt_(self: Self, other: Self) -> Self:
        return other._lt_(self)

    # le

    def __le__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._le_(b)
        assert result.shape == a.shape
        return result

    def _le_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'leing')

    # ge

    def __ge__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._ge_(b)
        assert result.shape == a.shape
        return result

    def _ge_(self: Self, other: Self) -> Self:
        return other._le_(self)

    # eq

    def __eq__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._eq_(b)
        assert result.shape == a.shape
        return result

    def _eq_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'eqing')


def MissingMethod(self, action) -> TypeError:
    return TypeError(f"No method for {action} objects of type {type(self)}.")

def MissingClassmethod(cls, action) -> TypeError:
    return TypeError(f"No classmethod for {action} of type {cls}.")


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
    # Ensure shape1 has higher rank.
    if rank1 < rank2:
        shape1, shape2 = shape2, shape1
        rank1, rank2 = rank2, rank1

    def broadcast_axis(axis) -> int:
        dim1 = shape1[axis]
        dim2 = shape2[axis]
        if dim1 == dim2:
            return dim1
        elif dim1 == 0 or dim2 == 0:
            raise ValueError(f"Cannot broadcast axis {axis} with size zero.")
        elif dim1 == 1:
            return dim2
        elif dim2 == 1:
            return dim1
        else:
            raise ValueError(f"Cannot broadcast axis {axis} with incompatible size.")

    return tuple(broadcast_axis(axis) for axis in range(rank2)) + shape1[rank2:]
