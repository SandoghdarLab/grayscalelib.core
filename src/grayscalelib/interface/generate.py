from __future__ import annotations
from itertools import product, permutations
from typing import Iterable

scalar_types = [
    "float",
]

# Methods that take one pixel container, and return one pixel container.
one_pix_methods = [
    "__abs__",
    "__not__",
    "__invert__",
    "__neg__",
    "__pos__",
]


# Methods that take two pixel containers or scalars, and return one pixel
# container whose shape is obtained via broadcasting.  Each entry of the
# following list is a triple whose first element is the name of the method, and
# whose second element is the name of the corresponding reverse method.
two_pix_methods = [
    ("__add__", "__radd__"),
    ("__and__", "__rand__"),
    ("__floordiv__", "__rfloordiv__"),
    ("__mod__", "__rmod__"),
    ("__mul__", "__rmul__"),
    ("__or__", "__ror__"),
    ("__sub__", "__rsub__"),
    ("__truediv__", "__rtruediv__"),
    ("__xor__", "__rxor__"),
    ("__lt__", None),
    ("__gt__", None),
    ("__le__", None),
    ("__ge__", None),
    ("__eq__", "__eq__"),
]

# Create overloads for all combinations of axes below this limit.
rank_limit = 5

axes = tuple(f"A{i}" for i in range(rank_limit))


def typestring(name: str, tokens: Iterable[str]):
    inner = ', '.join(tokens)
    if inner == '':
        inner = '()'
    return f"{name}[{inner}]"


def tupletype(axes):
    return typestring("tuple", axes)


def pixeltype(axes):
    return typestring("Pixels", axes)


def argtype(axes):
    if len(axes) == 0:
        return f"{pixeltype(axes)} | Scalar"
    else:
        return pixeltype(axes)


def write_file_header(file = None):
    def write(string, **kwargs):
        print(string, file=file, **kwargs)

    write("# This file was produced by generate.py.  Do not edit it by hand.\n")
    write("from __future__ import annotations\n")
    write("from abc import abstractmethod\n")
    write("from types import EllipsisType\n")
    write("from typing import Self, Generic, overload, Literal, TypeVar, TypeVarTuple\n")
    write("from encodable import choose_encoding, encode_as, Encodable\n")
    write("from grayscalelib.interface.auxiliary import *\n")
    for axis in axes:
        write(f"{axis} = TypeVar('{axis}')")
    write(f"Shape = TypeVarTuple('Shape')")
    write(f"Rest = TypeVarTuple('Rest')\n")
    write(f"Scalar = {' | '.join(scalar_types)}\n")
    write(f"fallback_encoding: type | None = None\n\n")
    write("def encoding(cls, *clss):")
    write("    if fallback_encoding is None:")
    write("        return choose_encoding(cls, *clss)")
    write("    else:")
    write("        return choose_encoding(cls, *clss, fallback_encoding)\n\n")


def write_class_header(file = None):
    def write(string, **kwargs):
        print(string, file=file, **kwargs)

    write("class Pixels(Encodable, Generic[*Shape]):\n")
    write("    @property")
    write("    @abstractmethod")
    write("    def shape(self) -> tuple[*Shape]: ...\n")
    write("    @property")
    write("    @abstractmethod")
    write("    def precision(self) -> int: ...\n")
    write("    @property")
    write("    def denominator(self) -> int:")
    write("        return 2 ** self.precision - 1\n")
    write("    @property")
    write("    def rank(self) -> int:")
    write("        return len(self.shape)\n")
    write("    def __len__(self: Pixels[A1, *tuple]) -> A1:")
    write("        if len(self.shape) == 0:")
    write("            raise RuntimeError(\"A rank zero container has no length.\")")
    write("        return self.shape[0]\n")
    write("    @classmethod")
    write("    def zeros(cls, shape: tuple[*Rest]) -> Pixels[*Rest]:")
    write("        result = cls._zeros_(shape)")
    write("        assert result.shape == shape")
    write("        assert result.precision == 0")
    write("        return result\n")
    write("    @classmethod")
    write("    def _zeros_(cls, shape: tuple[*Rest]) -> Pixels[*Rest]:")
    write("        _ = shape")
    write("        raise MissingClassmethod(cls, 'creating zeros')\n")
    write("    @classmethod")
    write("    def ones(cls, shape: tuple[*Rest]) -> Pixels[*Rest]:")
    write("        result = cls._ones_(shape)")
    write("        assert result.shape == shape")
    write("        assert result.precision == 1")
    write("        return result\n")
    write("    @classmethod")
    write("    def _ones_(cls, shape: tuple[*Rest]) -> Pixels[*Rest]:")
    write("        _ = shape")
    write("        raise MissingClassmethod(cls, 'creating ones')\n")


def write_getitem_methods(file = None):
    def write(string, **kwargs):
        print("    " + string, file=file, **kwargs)

    write("# getitem\n")
    for n in range(rank_limit):
        for index in product(('int', 'slice'), repeat=n):
            leftover = tuple(x for i, x in zip(index, axes) if i == 'slice')
            what = tupletype(index)
            before = argtype(axes[:n] + ('*Rest',))
            after = argtype(leftover + ('*Rest',))
            write("@overload")
            write(f"def __getitem__(self: {before}, index: {what}) -> {after}: ...\n")
    indextype = "EllipsisType | int | slice | tuple[int | slice, ...]"
    write(f"def __getitem__(self: Pixels, index: {indextype}) -> Pixels:")
    write(f"    return self._getitem_(canonicalize_index(index, self.shape))\n")
    write(f"def _getitem_(self, index: tuple[int | slice, ...]) -> Pixels[*tuple[int, ...]]:")
    write(f"    _ = index")
    write(f"    raise MissingMethod(self, 'indexing')\n")


def write_one_pix_methods(file = None):
    def write(string, **kwargs):
        print("    " + string, file=file, **kwargs)

    write("def __repr__(self) -> str:")
    write("""    return f"<{type(self).__name__} shape={self.shape} precision={self.precision}>"\n""")

    write("# bool\n")
    write("def __bool__(self) -> bool:")
    write("    cls = encoding(type(self))")
    write("    result = encode_as(self, cls)._bool_()")
    write("    assert result is True or result is False")
    write("    return result\n")
    write("def _bool_(self) -> bool:")
    write("    raise MissingMethod(self, 'determining the truth of')\n")

    write("# lshift\n")
    write("def __lshift__(self: Pixels[*Shape], amount: int) -> Pixels[*Shape]:")
    write("    if amount < 0:")
    write("        return self.__rshift__(-amount)")
    write("    cls = encoding(type(self))")
    write("    pix = encode_as(self, cls)")
    write("    result = pix._lshift_(amount)")
    write("    assert result.shape == self.shape")
    write("    assert result.precision == (self.precision + amount)")
    write("    return result\n")
    write("def _lshift_(self: Self, amount: int) -> Self:")
    write("    _ = amount")
    write("    raise MissingMethod(self, 'increasing the precision of')\n")

    write("# rshift\n")
    write("def __rshift__(self: Pixels[*Shape], amount: int) -> Pixels[*Shape]:")
    write("    if amount < 0:")
    write("        return self.__lshift__(-amount)")
    write("    cls = encoding(type(self))")
    write("    pix = encode_as(self, cls)")
    write("    result = pix._rshift_(amount)")
    write("    assert result.shape == self.shape")
    write("    assert result.precision == max(0, (self.precision - amount))")
    write("    return result\n")
    write("def _rshift_(self: Self, amount: int) -> Self:")
    write("    _ = amount")
    write("    raise MissingMethod(self, 'decreasing the precision of')\n")

    write("# pow\n")
    write("def __pow__(self: Pixels[*Shape], exponent: float) -> Pixels[*Shape]:")
    write("    cls = encoding(type(self))")
    write("    result = encode_as(self, cls)._pow_(exponent)")
    write("    assert result.shape == self.shape")
    write("    return result\n")
    write("def _pow_(self: Self, exponent: float) -> Self:")
    write("    _ = exponent")
    write("    raise MissingMethod(self, 'exponentiating')\n")

    for method in one_pix_methods:
        write(f"# {method[2:-2]}\n")
        write(f"def {method}(self: Pixels[*Shape]) -> Pixels[*Shape]:")
        write(f"    cls = encoding(type(self))")
        write(f"    result = encode_as(self, cls).{method[1:-1]}")
        write(f"    assert result.shape == self.shape")
        write(f"    return result\n")
        write(f"def {method[1:-1]}(self: Self) -> Self:")
        write(f"    raise MissingMethod(self, '{method[2:-2]}ing')\n")


def write_two_pix_methods(file = None):
    def write(string, **kwargs):
        print("    " + string, file=file, **kwargs)

    write("# broadcast_to\n")
    write("def broadcast_to(self, shape: tuple[*Rest]) -> Pixels[*Rest]:")
    write("    result = self._broadcast_to_(shape)")
    write("    assert result.shape == shape")
    write("    return result\n")
    write("def _broadcast_to_(self, shape: tuple[*Rest]) -> Pixels[*Rest]:")
    write("    _ = shape")
    write("    raise MissingMethod(self, 'broadcasting')\n")

    write(f"# broadcast\n")
    t = "Pixels[*Shape]"
    write("@overload")
    write(f"def __broadcast__(self: {t}, other: Scalar) -> tuple[{t}, {t}]: ...\n")
    t = "Pixels[*Shape, A1, *Rest]"
    write("@overload")
    write(f"def __broadcast__(self: {t}, other: Pixels[*Shape]) -> tuple[{t}, {t}]: ...\n")
    write("@overload")
    write(f"def __broadcast__(self: Pixels[*Shape], other: {t}) -> tuple[{t}, {t}]: ...\n")
    write(f"def __broadcast__(self: Pixels, other: Pixels | Scalar) -> tuple[Pixels, Pixels]:")
    write(f"    cls = encoding(type(self), type(other))")
    write(f"    a = encode_as(self, cls)")
    write(f"    b = encode_as(other, cls)")
    write(f"    s = broadcast_shapes(a.shape, b.shape)")
    write(f"    return (a.broadcast_to(s), b.broadcast_to(s))\n")

    for (method, rmethod) in two_pix_methods:
        write(f"# {method[2:-2]}\n")
        t = "Pixels[*Shape, A0, *Rest]"
        write("@overload")
        write(f"def {method}(self: {t}, other: Pixels[*Shape]) -> {t}: ...\n")
        write("@overload")
        write(f"def {method}(self: Pixels[*Shape], other: {t}) -> {t}: ...\n")
        t = "Pixels[*Shape]"
        write("@overload")
        write(f"def {method}(self: {t}, other: Scalar) -> {t}: ...\n")
        write(f"def {method}(self: Pixels, other: Pixels | Scalar) -> Pixels:")
        write(f"    a, b = self.__broadcast__(other)")
        write(f"    result = a.{method[1:-1]}(b)")
        write(f"    assert result.shape == a.shape")
        write(f"    return result\n")
        if rmethod and method != rmethod:
            write(f"def {rmethod}(self: {t}, other: Scalar) -> {t}:")
            write(f"    b, a = self.__broadcast__(other)")
            write(f"    return a.{method[1:-1]}(b)\n")
        if method == "__gt__":
            write("def _gt_(self: Self, other: Self) -> Self:")
            write("    return other._lt_(self)\n")
        elif method == "__ge__":
            write("def _ge_(self: Self, other: Self) -> Self:")
            write("    return other._le_(self)\n")
        else:
            write(f"def {method[1:-1]}(self: Self, other: Self) -> Self:")
            write(f"    _ = other")
            write(f"    raise MissingMethod(self, '{method[2:-2]}ing')\n")


def write_permute_methods(file = None):
    def write(string, **kwargs):
        print("    " + string, file=file, **kwargs)

    def indextype(n: int):
        if n == -1:
            return "int"
        else:
            return f"Literal[{n}]"

    variants: set[tuple[int, ...]] = set()
    for rank in range(rank_limit):
        for permutation in permutations(range(rank)):
            for variant in product(*tuple((p, -1) for p in permutation)):
                if len(variant) == 0 or variant.count(-1) >= 1:
                    variants.add(variant)

    write(f"# permute\n")
    for variant in sorted(variants, key=lambda v: rank_limit*len(v) + v.count(-1)):
        rank = len(variant)
        selftype = pixeltype(axes[:rank])
        vague = " | ".join(axes[i] for i in range(rank) if i not in variant)
        result = pixeltype(axes[p] if p >= 0 else vague for p in variant)
        signature = "".join(f", i{i}: {indextype(p)}" for i, p in enumerate(variant)) + ", /"
        write("@overload")
        write(f"def permute(self: {selftype}{signature}) -> {result}: ...\n")
    write(f"def permute(self, *permutation: int) -> Pixels:")
    write(f"    permute_check_before(permutation, self.rank)")
    write(f"    cls = encoding(type(self))")
    write(f"    result = encode_as(self, cls)._permute_(permutation)")
    write(f"    permute_check_after(permutation, self.shape, result.shape)")
    write(f"    return result\n")
    write(f"def _permute_(self, permutation: tuple[int, ...]) -> Pixels:")
    write(f"    _ = permutation")
    write(f"    raise MissingMethod(self, 'permuting')\n")


if __name__ == "__main__":
    with open("pixels.py", "w") as f:
        write_file_header(f)
        write_class_header(f)
        write_getitem_methods(f)
        write_permute_methods(f)
        write_one_pix_methods(f)
        write_two_pix_methods(f)
