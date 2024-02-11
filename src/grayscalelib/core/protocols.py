from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArrayLike(Protocol):
    """
    Protocol class for things that look like an array.
    """
    def __getitem__(self, key) -> Any:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...


@runtime_checkable
class RealLike(Protocol):
    """
    Protocol class for things that look like a real number.
    """
    def __float__(self) -> float:
        ...

    def __round__(self) -> int:
        ...

    def __floor__(self) -> int:
        ...

    def __ceil__(self) -> int:
        ...

    def __trunc__(self) -> int:
        ...

    def __add__(self, other, /) -> RealLike:
        ...

    def __radd__(self, other, /) -> RealLike:
        ...

    def __mul__(self, other, /) -> RealLike:
        ...

    def __rmul__(self, other, /) -> RealLike:
        ...

    def __sub__(self, other, /) -> RealLike:
        ...

    def __rsub__(self, other, /) -> RealLike:
        ...

    def __truediv__(self, other, /) -> RealLike:
        ...

    def __rtruediv__(self, other, /) -> RealLike:
        ...

    def __floordiv__(self, other, /) -> RealLike:
        ...

    def __rfloordiv__(self, other, /) -> RealLike:
        ...

    def __mod__(self, other, /) -> RealLike:
        ...

    def __rmod__(self, other, /) -> RealLike:
        ...

    def __lt__(self, other, /) -> bool:
        ...

    def __le__(self, other, /) -> bool:
        ...

    def __gt__(self, other, /) -> bool:
        ...

    def __ge__(self, other, /) -> bool:
        ...

    def __eq__(self, other, /) -> bool:
        ...
