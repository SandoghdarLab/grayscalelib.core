# This file was produced by generate.py.  Do not edit it by hand.

from __future__ import annotations

from abc import abstractmethod

from types import EllipsisType

from typing import Self, Generic, overload, Literal, TypeVar, TypeVarTuple

from encodable import choose_encoding, encode_as, Encodable

from grayscalelib.interface.auxiliary import *

A0 = TypeVar('A0')
A1 = TypeVar('A1')
A2 = TypeVar('A2')
A3 = TypeVar('A3')
A4 = TypeVar('A4')
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

    def __len__(self: Pixels[A1, *tuple]) -> A1:
        if len(self.shape) == 0:
            raise RuntimeError("A rank zero container has no length.")
        return self.shape[0]

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

    @overload
    def __getitem__(self: Pixels[*Rest], index: tuple[()]) -> Pixels[*Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, *Rest], index: tuple[int]) -> Pixels[*Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, *Rest], index: tuple[slice]) -> Pixels[A0, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, *Rest], index: tuple[int, int]) -> Pixels[*Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, *Rest], index: tuple[int, slice]) -> Pixels[A1, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, *Rest], index: tuple[slice, int]) -> Pixels[A0, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, *Rest], index: tuple[slice, slice]) -> Pixels[A0, A1, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, *Rest], index: tuple[int, int, int]) -> Pixels[*Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, *Rest], index: tuple[int, int, slice]) -> Pixels[A2, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, *Rest], index: tuple[int, slice, int]) -> Pixels[A1, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, *Rest], index: tuple[int, slice, slice]) -> Pixels[A1, A2, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, *Rest], index: tuple[slice, int, int]) -> Pixels[A0, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, *Rest], index: tuple[slice, int, slice]) -> Pixels[A0, A2, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, *Rest], index: tuple[slice, slice, int]) -> Pixels[A0, A1, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, *Rest], index: tuple[slice, slice, slice]) -> Pixels[A0, A1, A2, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[int, int, int, int]) -> Pixels[*Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[int, int, int, slice]) -> Pixels[A3, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[int, int, slice, int]) -> Pixels[A2, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[int, int, slice, slice]) -> Pixels[A2, A3, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[int, slice, int, int]) -> Pixels[A1, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[int, slice, int, slice]) -> Pixels[A1, A3, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[int, slice, slice, int]) -> Pixels[A1, A2, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[int, slice, slice, slice]) -> Pixels[A1, A2, A3, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[slice, int, int, int]) -> Pixels[A0, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[slice, int, int, slice]) -> Pixels[A0, A3, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[slice, int, slice, int]) -> Pixels[A0, A2, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[slice, int, slice, slice]) -> Pixels[A0, A2, A3, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[slice, slice, int, int]) -> Pixels[A0, A1, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[slice, slice, int, slice]) -> Pixels[A0, A1, A3, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[slice, slice, slice, int]) -> Pixels[A0, A1, A2, *Rest]: ...

    @overload
    def __getitem__(self: Pixels[A0, A1, A2, A3, *Rest], index: tuple[slice, slice, slice, slice]) -> Pixels[A0, A1, A2, A3, *Rest]: ...

    def __getitem__(self: Pixels, index: EllipsisType | int | slice | tuple[int | slice, ...]) -> Pixels:
        return self._getitem_(canonicalize_index(index, self.shape))

    def _getitem_(self, index: tuple[int | slice, ...]) -> Pixels[*tuple[int, ...]]:
        _ = index
        raise MissingMethod(self, 'indexing')

    # permute

    @overload
    def permute(self: Pixels[()], /) -> Pixels[()]: ...

    @overload
    def permute(self: Pixels[A0], i0: int, /) -> Pixels[A0]: ...

    @overload
    def permute(self: Pixels[A0, A1], i0: Literal[0], i1: int, /) -> Pixels[A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1], i0: int, i1: Literal[0], /) -> Pixels[A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1], i0: Literal[1], i1: int, /) -> Pixels[A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1], i0: int, i1: Literal[1], /) -> Pixels[A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1], i0: int, i1: int, /) -> Pixels[A0 | A1, A0 | A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[2], i1: int, i2: Literal[1], /) -> Pixels[A2, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[0], i1: int, i2: Literal[2], /) -> Pixels[A0, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: Literal[0], i2: Literal[2], /) -> Pixels[A1, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[2], i1: Literal[0], i2: int, /) -> Pixels[A2, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[2], i1: int, i2: Literal[0], /) -> Pixels[A2, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: Literal[2], i2: Literal[1], /) -> Pixels[A0, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[0], i1: Literal[1], i2: int, /) -> Pixels[A0, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[0], i1: int, i2: Literal[1], /) -> Pixels[A0, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[0], i1: Literal[2], i2: int, /) -> Pixels[A0, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[1], i1: Literal[0], i2: int, /) -> Pixels[A1, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[1], i1: int, i2: Literal[0], /) -> Pixels[A1, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[1], i1: Literal[2], i2: int, /) -> Pixels[A1, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: Literal[0], i2: Literal[1], /) -> Pixels[A2, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[2], i1: Literal[1], i2: int, /) -> Pixels[A2, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[1], i1: int, i2: Literal[2], /) -> Pixels[A1, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: Literal[1], i2: Literal[0], /) -> Pixels[A2, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: Literal[1], i2: Literal[2], /) -> Pixels[A0, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: Literal[2], i2: Literal[0], /) -> Pixels[A1, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: Literal[0], i2: int, /) -> Pixels[A1 | A2, A0, A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: int, i2: Literal[0], /) -> Pixels[A1 | A2, A1 | A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: int, i2: Literal[2], /) -> Pixels[A0 | A1, A0 | A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[2], i1: int, i2: int, /) -> Pixels[A2, A0 | A1, A0 | A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: Literal[1], i2: int, /) -> Pixels[A0 | A2, A1, A0 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[1], i1: int, i2: int, /) -> Pixels[A1, A0 | A2, A0 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: int, i2: Literal[1], /) -> Pixels[A0 | A2, A0 | A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: Literal[2], i2: int, /) -> Pixels[A0 | A1, A2, A0 | A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: Literal[0], i1: int, i2: int, /) -> Pixels[A0, A1 | A2, A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2], i0: int, i1: int, i2: int, /) -> Pixels[A0 | A1 | A2, A0 | A1 | A2, A0 | A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: Literal[0], i3: Literal[3], /) -> Pixels[A2, A1, A0, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[3], i2: int, i3: Literal[2], /) -> Pixels[A0, A3, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: Literal[0], i3: Literal[2], /) -> Pixels[A3, A1, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: Literal[2], i3: Literal[0], /) -> Pixels[A3, A1, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: Literal[0], i3: Literal[1], /) -> Pixels[A3, A2, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[2], i2: Literal[3], i3: int, /) -> Pixels[A1, A2, A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[1], i2: Literal[2], i3: int, /) -> Pixels[A3, A1, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[2], i2: Literal[0], i3: int, /) -> Pixels[A3, A2, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: Literal[1], i3: Literal[2], /) -> Pixels[A0, A3, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: Literal[1], i3: Literal[0], /) -> Pixels[A3, A2, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: Literal[2], i3: Literal[3], /) -> Pixels[A0, A1, A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[2], i2: int, i3: Literal[0], /) -> Pixels[A3, A2, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: Literal[2], i3: Literal[1], /) -> Pixels[A3, A0, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[1], i2: int, i3: Literal[0], /) -> Pixels[A3, A1, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: Literal[0], i3: Literal[3], /) -> Pixels[A1, A2, A0, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[0], i2: Literal[3], i3: int, /) -> Pixels[A1, A0, A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[0], i2: int, i3: Literal[2], /) -> Pixels[A1, A0, A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: Literal[1], i3: Literal[2], /) -> Pixels[A3, A0, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[1], i2: int, i3: Literal[3], /) -> Pixels[A0, A1, A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[3], i2: Literal[0], i3: int, /) -> Pixels[A1, A3, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[2], i2: int, i3: Literal[1], /) -> Pixels[A0, A2, A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[2], i2: Literal[1], i3: int, /) -> Pixels[A3, A2, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[1], i2: int, i3: Literal[2], /) -> Pixels[A3, A1, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[3], i2: Literal[2], i3: int, /) -> Pixels[A1, A3, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: Literal[0], i3: Literal[1], /) -> Pixels[A2, A3, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: Literal[0], i3: Literal[2], /) -> Pixels[A1, A3, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[3], i2: Literal[1], i3: int, /) -> Pixels[A0, A3, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[2], i2: int, i3: Literal[3], /) -> Pixels[A0, A2, A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[0], i2: int, i3: Literal[3], /) -> Pixels[A2, A0, A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[1], i2: Literal[0], i3: int, /) -> Pixels[A3, A1, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: Literal[0], i3: Literal[1], /) -> Pixels[A2, A3, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: Literal[3], i3: Literal[0], /) -> Pixels[A1, A2, A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[1], i2: Literal[2], i3: int, /) -> Pixels[A0, A1, A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: Literal[0], i3: Literal[2], /) -> Pixels[A3, A1, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[1], i2: Literal[3], i3: int, /) -> Pixels[A2, A1, A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: Literal[0], i3: Literal[3], /) -> Pixels[A2, A1, A0, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: Literal[3], i3: Literal[1], /) -> Pixels[A0, A2, A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: Literal[2], i3: Literal[3], /) -> Pixels[A1, A0, A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: Literal[3], i3: Literal[2], /) -> Pixels[A1, A0, A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: Literal[3], i3: Literal[0], /) -> Pixels[A2, A1, A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: Literal[3], i3: Literal[1], /) -> Pixels[A0, A2, A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[0], i2: int, i3: Literal[2], /) -> Pixels[A3, A0, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: Literal[3], i3: Literal[2], /) -> Pixels[A0, A1, A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[3], i2: Literal[1], i3: int, /) -> Pixels[A2, A3, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[0], i2: Literal[1], i3: int, /) -> Pixels[A3, A0, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: Literal[1], i3: Literal[3], /) -> Pixels[A0, A2, A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: Literal[2], i3: Literal[1], /) -> Pixels[A3, A0, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: Literal[2], i3: Literal[0], /) -> Pixels[A1, A3, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: Literal[1], i3: Literal[2], /) -> Pixels[A3, A0, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[3], i2: int, i3: Literal[1], /) -> Pixels[A2, A3, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[3], i2: int, i3: Literal[1], /) -> Pixels[A0, A3, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: Literal[0], i3: Literal[1], /) -> Pixels[A3, A2, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: Literal[2], i3: Literal[3], /) -> Pixels[A1, A0, A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: Literal[3], i3: Literal[2], /) -> Pixels[A1, A0, A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[2], i2: Literal[0], i3: int, /) -> Pixels[A1, A2, A0, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: Literal[0], i3: Literal[3], /) -> Pixels[A1, A2, A0, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: Literal[2], i3: Literal[0], /) -> Pixels[A1, A3, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: Literal[3], i3: Literal[1], /) -> Pixels[A2, A0, A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: Literal[2], i3: Literal[0], /) -> Pixels[A3, A1, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: Literal[1], i3: Literal[3], /) -> Pixels[A2, A0, A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: Literal[2], i3: Literal[1], /) -> Pixels[A0, A3, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: Literal[1], i3: Literal[3], /) -> Pixels[A0, A2, A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[2], i2: int, i3: Literal[3], /) -> Pixels[A1, A2, A0, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[1], i2: int, i3: Literal[2], /) -> Pixels[A0, A1, A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[1], i2: Literal[3], i3: int, /) -> Pixels[A0, A1, A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[3], i2: int, i3: Literal[0], /) -> Pixels[A1, A3, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[1], i2: int, i3: Literal[3], /) -> Pixels[A2, A1, A0, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[2], i2: int, i3: Literal[1], /) -> Pixels[A3, A2, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: Literal[1], i3: Literal[0], /) -> Pixels[A3, A2, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[0], i2: Literal[2], i3: int, /) -> Pixels[A1, A0, A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: Literal[2], i3: Literal[3], /) -> Pixels[A0, A1, A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[2], i2: Literal[1], i3: int, /) -> Pixels[A0, A2, A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[0], i2: Literal[1], i3: int, /) -> Pixels[A2, A0, A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[0], i2: int, i3: Literal[3], /) -> Pixels[A1, A0, A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: Literal[3], i3: Literal[2], /) -> Pixels[A0, A1, A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: Literal[3], i3: Literal[0], /) -> Pixels[A1, A2, A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[3], i2: int, i3: Literal[2], /) -> Pixels[A1, A3, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[0], i2: int, i3: Literal[1], /) -> Pixels[A3, A0, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[2], i2: Literal[3], i3: int, /) -> Pixels[A0, A2, A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[0], i2: Literal[3], i3: int, /) -> Pixels[A2, A0, A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[3], i2: Literal[2], i3: int, /) -> Pixels[A0, A3, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: Literal[0], i3: Literal[2], /) -> Pixels[A1, A3, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[0], i2: Literal[2], i3: int, /) -> Pixels[A3, A0, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[3], i2: Literal[0], i3: int, /) -> Pixels[A2, A3, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[0], i2: int, i3: Literal[1], /) -> Pixels[A2, A0, A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: Literal[1], i3: Literal[0], /) -> Pixels[A2, A3, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[3], i2: int, i3: Literal[0], /) -> Pixels[A2, A3, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: Literal[3], i3: Literal[1], /) -> Pixels[A2, A0, A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: Literal[1], i3: Literal[0], /) -> Pixels[A2, A3, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: Literal[2], i3: Literal[1], /) -> Pixels[A0, A3, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[2], i2: int, i3: Literal[0], /) -> Pixels[A1, A2, A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[1], i2: Literal[0], i3: int, /) -> Pixels[A2, A1, A0, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: Literal[1], i3: Literal[3], /) -> Pixels[A2, A0, A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[1], i2: int, i3: Literal[0], /) -> Pixels[A2, A1, A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: Literal[3], i3: Literal[0], /) -> Pixels[A2, A1, A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: Literal[1], i3: Literal[2], /) -> Pixels[A0, A3, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[1], i2: int, i3: int, /) -> Pixels[A3, A1, A0 | A2, A0 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[3], i3: Literal[1], /) -> Pixels[A0 | A2, A0 | A2, A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: int, i3: Literal[3], /) -> Pixels[A0, A1 | A2, A1 | A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: int, i3: Literal[1], /) -> Pixels[A0 | A3, A2, A0 | A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[1], i3: Literal[3], /) -> Pixels[A0 | A2, A0 | A2, A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: int, i3: Literal[2], /) -> Pixels[A0 | A3, A1, A0 | A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[0], i2: int, i3: int, /) -> Pixels[A3, A0, A1 | A2, A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: int, i3: Literal[3], /) -> Pixels[A0 | A1, A2, A0 | A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: Literal[1], i3: int, /) -> Pixels[A2 | A3, A0, A1, A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: Literal[0], i3: int, /) -> Pixels[A2 | A3, A1, A0, A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: Literal[3], i3: int, /) -> Pixels[A1 | A2, A0, A3, A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[2], i3: Literal[0], /) -> Pixels[A1 | A3, A1 | A3, A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: Literal[2], i3: int, /) -> Pixels[A0 | A1, A3, A2, A0 | A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: Literal[2], i3: int, /) -> Pixels[A0, A1 | A3, A2, A1 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[0], i3: Literal[2], /) -> Pixels[A1 | A3, A1 | A3, A0, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: int, i3: Literal[1], /) -> Pixels[A2, A0 | A3, A0 | A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: Literal[0], i3: int, /) -> Pixels[A1 | A3, A2, A0, A1 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: int, i3: Literal[1], /) -> Pixels[A3, A0 | A2, A0 | A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[1], i2: int, i3: int, /) -> Pixels[A0, A1, A2 | A3, A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: int, i3: Literal[3], /) -> Pixels[A2, A0 | A1, A0 | A1, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: Literal[3], i3: int, /) -> Pixels[A0 | A2, A1, A3, A0 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: int, i3: Literal[2], /) -> Pixels[A0, A1 | A3, A1 | A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: int, i3: Literal[0], /) -> Pixels[A1 | A3, A2, A1 | A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[2], i2: int, i3: int, /) -> Pixels[A0, A2, A1 | A3, A1 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: int, i3: Literal[2], /) -> Pixels[A1 | A3, A0, A1 | A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[3], i2: int, i3: int, /) -> Pixels[A1, A3, A0 | A2, A0 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[0], i2: int, i3: int, /) -> Pixels[A2, A0, A1 | A3, A1 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: Literal[0], i3: int, /) -> Pixels[A2, A1 | A3, A0, A1 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: int, i3: Literal[0], /) -> Pixels[A1 | A2, A3, A1 | A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: Literal[2], i3: int, /) -> Pixels[A1, A0 | A3, A2, A0 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[0], i3: Literal[1], /) -> Pixels[A2 | A3, A2 | A3, A0, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: int, i3: Literal[0], /) -> Pixels[A2, A1 | A3, A1 | A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[1], i3: Literal[0], /) -> Pixels[A2 | A3, A2 | A3, A1, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: Literal[1], i3: int, /) -> Pixels[A0 | A2, A3, A1, A0 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: int, i3: Literal[3], /) -> Pixels[A1, A0 | A2, A0 | A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[2], i3: Literal[1], /) -> Pixels[A0 | A3, A0 | A3, A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: Literal[2], i3: int, /) -> Pixels[A3, A0 | A1, A2, A0 | A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[3], i3: Literal[0], /) -> Pixels[A1 | A2, A1 | A2, A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: int, i3: Literal[2], /) -> Pixels[A0 | A1, A3, A0 | A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: Literal[3], i3: int, /) -> Pixels[A0, A1 | A2, A3, A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[1], i3: Literal[2], /) -> Pixels[A0 | A3, A0 | A3, A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[0], i3: Literal[3], /) -> Pixels[A1 | A2, A1 | A2, A0, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: Literal[1], i3: int, /) -> Pixels[A0 | A3, A2, A1, A0 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[2], i3: Literal[3], /) -> Pixels[A0 | A1, A0 | A1, A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[3], i3: Literal[2], /) -> Pixels[A0 | A1, A0 | A1, A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: Literal[0], i3: int, /) -> Pixels[A1 | A2, A3, A0, A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: Literal[3], i3: int, /) -> Pixels[A0 | A1, A2, A3, A0 | A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: int, i3: Literal[1], /) -> Pixels[A2 | A3, A0, A2 | A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: int, i3: Literal[3], /) -> Pixels[A0 | A2, A1, A0 | A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: Literal[2], i3: int, /) -> Pixels[A1 | A3, A0, A2, A1 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: Literal[0], i3: int, /) -> Pixels[A1, A2 | A3, A0, A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: int, i3: Literal[3], /) -> Pixels[A1 | A2, A0, A1 | A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: int, i3: Literal[0], /) -> Pixels[A1, A2 | A3, A2 | A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: Literal[3], i2: int, i3: int, /) -> Pixels[A0, A3, A1 | A2, A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[3], i2: int, i3: int, /) -> Pixels[A2, A3, A0 | A1, A0 | A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: Literal[0], i3: int, /) -> Pixels[A3, A1 | A2, A0, A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: Literal[1], i3: int, /) -> Pixels[A2, A0 | A3, A1, A0 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: Literal[1], i3: int, /) -> Pixels[A0, A2 | A3, A1, A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[2], i2: int, i3: int, /) -> Pixels[A1, A2, A0 | A3, A0 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: int, i3: Literal[0], /) -> Pixels[A3, A1 | A2, A1 | A2, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: Literal[1], i2: int, i3: int, /) -> Pixels[A2, A1, A0 | A3, A0 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: Literal[3], i3: int, /) -> Pixels[A2, A0 | A1, A3, A0 | A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: int, i3: Literal[2], /) -> Pixels[A1, A0 | A3, A0 | A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: Literal[2], i3: int, /) -> Pixels[A0 | A3, A1, A2, A0 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: int, i3: Literal[1], /) -> Pixels[A0 | A2, A3, A0 | A2, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: int, i3: Literal[1], /) -> Pixels[A0, A2 | A3, A2 | A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: Literal[0], i2: int, i3: int, /) -> Pixels[A1, A0, A2 | A3, A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: int, i3: Literal[2], /) -> Pixels[A3, A0 | A1, A0 | A1, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: Literal[3], i3: int, /) -> Pixels[A1, A0 | A2, A3, A0 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: Literal[1], i3: int, /) -> Pixels[A3, A0 | A2, A1, A0 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: int, i3: Literal[0], /) -> Pixels[A2 | A3, A1, A2 | A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: Literal[2], i2: int, i3: int, /) -> Pixels[A3, A2, A0 | A1, A0 | A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: int, i3: Literal[0], /) -> Pixels[A1 | A2 | A3, A1 | A2 | A3, A1 | A2 | A3, A0]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[0], i1: int, i2: int, i3: int, /) -> Pixels[A0, A1 | A2 | A3, A1 | A2 | A3, A1 | A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[1], i3: int, /) -> Pixels[A0 | A2 | A3, A0 | A2 | A3, A1, A0 | A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: int, i3: Literal[2], /) -> Pixels[A0 | A1 | A3, A0 | A1 | A3, A0 | A1 | A3, A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[2], i2: int, i3: int, /) -> Pixels[A0 | A1 | A3, A2, A0 | A1 | A3, A0 | A1 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[3], i2: int, i3: int, /) -> Pixels[A0 | A1 | A2, A3, A0 | A1 | A2, A0 | A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[2], i1: int, i2: int, i3: int, /) -> Pixels[A2, A0 | A1 | A3, A0 | A1 | A3, A0 | A1 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[3], i3: int, /) -> Pixels[A0 | A1 | A2, A0 | A1 | A2, A3, A0 | A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: int, i3: Literal[1], /) -> Pixels[A0 | A2 | A3, A0 | A2 | A3, A0 | A2 | A3, A1]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[1], i1: int, i2: int, i3: int, /) -> Pixels[A1, A0 | A2 | A3, A0 | A2 | A3, A0 | A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[0], i3: int, /) -> Pixels[A1 | A2 | A3, A1 | A2 | A3, A0, A1 | A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: Literal[3], i1: int, i2: int, i3: int, /) -> Pixels[A3, A0 | A1 | A2, A0 | A1 | A2, A0 | A1 | A2]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: Literal[2], i3: int, /) -> Pixels[A0 | A1 | A3, A0 | A1 | A3, A2, A0 | A1 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[1], i2: int, i3: int, /) -> Pixels[A0 | A2 | A3, A1, A0 | A2 | A3, A0 | A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: int, i3: Literal[3], /) -> Pixels[A0 | A1 | A2, A0 | A1 | A2, A0 | A1 | A2, A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: Literal[0], i2: int, i3: int, /) -> Pixels[A1 | A2 | A3, A0, A1 | A2 | A3, A1 | A2 | A3]: ...

    @overload
    def permute(self: Pixels[A0, A1, A2, A3], i0: int, i1: int, i2: int, i3: int, /) -> Pixels[A0 | A1 | A2 | A3, A0 | A1 | A2 | A3, A0 | A1 | A2 | A3, A0 | A1 | A2 | A3]: ...

    def permute(self, *permutation: int) -> Pixels:
        permute_check_before(permutation, self.rank)
        cls = encoding(type(self))
        result = encode_as(self, cls)._permute_(permutation)
        permute_check_after(permutation, self.shape, result.shape)
        return result

    def _permute_(self, permutation: tuple[int, ...]) -> Pixels:
        _ = permutation
        raise MissingMethod(self, 'permuting')

    def __repr__(self) -> str:
        return f"<{type(self).__name__} shape={self.shape} precision={self.precision}>"

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

    @overload
    def __broadcast__(self: Pixels[*Shape], other: Scalar) -> tuple[Pixels[*Shape], Pixels[*Shape]]: ...

    @overload
    def __broadcast__(self: Pixels[*Shape, A1, *Rest], other: Pixels[*Shape]) -> tuple[Pixels[*Shape, A1, *Rest], Pixels[*Shape, A1, *Rest]]: ...

    @overload
    def __broadcast__(self: Pixels[*Shape], other: Pixels[*Shape, A1, *Rest]) -> tuple[Pixels[*Shape, A1, *Rest], Pixels[*Shape, A1, *Rest]]: ...

    def __broadcast__(self: Pixels, other: Pixels | Scalar) -> tuple[Pixels, Pixels]:
        cls = encoding(type(self), type(other))
        a = encode_as(self, cls)
        b = encode_as(other, cls)
        s = broadcast_shapes(a.shape, b.shape)
        return (a.broadcast_to(s), b.broadcast_to(s))

    # add

    @overload
    def __add__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __add__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __add__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

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

    @overload
    def __and__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __and__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __and__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

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

    @overload
    def __floordiv__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __floordiv__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __floordiv__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

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

    @overload
    def __mod__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __mod__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __mod__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

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

    @overload
    def __mul__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __mul__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __mul__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

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

    @overload
    def __or__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __or__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __or__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

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

    @overload
    def __sub__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __sub__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __sub__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

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

    @overload
    def __truediv__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __truediv__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __truediv__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

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

    @overload
    def __xor__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __xor__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __xor__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

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

    @overload
    def __lt__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __lt__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __lt__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

    def __lt__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._lt_(b)
        assert result.shape == a.shape
        return result

    def _lt_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'lting')

    # gt

    @overload
    def __gt__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __gt__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __gt__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

    def __gt__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._gt_(b)
        assert result.shape == a.shape
        return result

    def _gt_(self: Self, other: Self) -> Self:
        return other._lt_(self)

    # le

    @overload
    def __le__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __le__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __le__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

    def __le__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._le_(b)
        assert result.shape == a.shape
        return result

    def _le_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'leing')

    # ge

    @overload
    def __ge__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __ge__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __ge__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

    def __ge__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._ge_(b)
        assert result.shape == a.shape
        return result

    def _ge_(self: Self, other: Self) -> Self:
        return other._le_(self)

    # eq

    @overload
    def __eq__(self: Pixels[*Shape, A0, *Rest], other: Pixels[*Shape]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __eq__(self: Pixels[*Shape], other: Pixels[*Shape, A0, *Rest]) -> Pixels[*Shape, A0, *Rest]: ...

    @overload
    def __eq__(self: Pixels[*Shape], other: Scalar) -> Pixels[*Shape]: ...

    def __eq__(self: Pixels, other: Pixels | Scalar) -> Pixels:
        a, b = self.__broadcast__(other)
        result = a._eq_(b)
        assert result.shape == a.shape
        return result

    def _eq_(self: Self, other: Self) -> Self:
        _ = other
        raise MissingMethod(self, 'eqing')

