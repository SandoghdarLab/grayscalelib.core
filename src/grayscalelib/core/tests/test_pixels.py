from typing import Callable
import pytest
import itertools
import math
from fractions import Fraction
from itertools import chain, permutations, product
from grayscalelib.core.pixels import Pixels, pixels_type
from grayscalelib.core.simplearray import SimpleArray


###############################################################################
###
### Fractional Math


Frac = int | Fraction


def frac(a: int, b: int) -> Frac:
    if b == 1:
        return a
    else:
        return Fraction(a, b)


def fracround(f: Frac, fbits: int) -> Frac:
    """
    Approximate f using a rational number with a denominator of 2**fbits.
    If two rational numbers are equally close, pick the larger one.
    """
    denominator = 1 << fbits
    eps = Fraction(1, denominator << 1)
    return frac(math.floor((f+eps) * denominator), denominator)


def clip(x, lo, hi):
    return max(lo, min(x, hi))


def strip_bits(i: int, n: int):
    """
    Divide the integer i by 2**n and round to the nearest integer.  In
    ambiguous cases, round to the next larger integer.
    """
    if n == 0:
        return i
    elif n > 0:
        return (i + (1 << (n - 1))) >> n
    else:
        raise ValueError("Negative number of stripped digits.")


###############################################################################
###
### Tests


@pytest.fixture
def pixels_subclass(request):
    assert issubclass(request.param, Pixels)
    with pixels_type(request.param):
        yield request.param


def test_init(pixels_subclass):
    # Check some trivial cases.
    assert Pixels(0, white=0, fbits=0).to_array()[()] == 0
    assert Pixels(1, fbits=0).to_array()[()] == 1
    assert Pixels(0.5, fbits=0).to_array()[()] == 1
    # Ensure that shapes are computed correctly.
    assert Pixels(0).shape == ()
    assert Pixels([]).shape == (0,)
    assert Pixels([[]]).shape == (1, 0)
    assert Pixels([[[]]]).shape == (1, 1, 0)
    assert Pixels([[0], [1]]).shape == (2, 1)
    # When converting real numbers in the [0, 1] range to pixels, the resulting
    # round-off error should be at most 2**(-fbits-1).
    for fbits in range(14):
        maxerr = 2**(-fbits-1)
        denominator = 2**fbits
        size = denominator+1
        data = [n / denominator for n in range(size)]
        px = Pixels(data, fbits=fbits)
        assert isinstance(px, pixels_subclass)
        array = px.to_array()
        for index in range(size):
            assert abs(array[index] - data[index]) < maxerr
    # Check that black, white, and limit are correctly accounted for.
    for fbits, black in product([0, 1, 2], range(-3, 3)):
        for white, limit in product(range(black, black+3), range(black, black+9)):
            if black == white and white != limit:
                continue
            delta = (white - black)
            data = [[black-1, black, black+1],
                    [white-1, white, white+1],
                    [limit-1, limit, limit+1]]
            px = Pixels(data, fbits=fbits, black=black, white=white, limit=limit)
            numerators = px.numerators
            for i, j in product(*tuple(range(s) for s in px.shape)):
                expected = black if delta == 0 else max(black, min(data[i][j], limit))
                restored = ((numerators[i, j] * delta) >> fbits) + black
                assert black <= restored <= limit + delta / 2
                assert abs(restored - expected) <= delta
    # Ensure that the ibits are computed correctly.
    for white in range(1, 5):
        for limit in range(white, 17):
            data = list(range(-1, limit+1))
            px = Pixels(data, fbits=0, white=white, limit=limit)
            bound = math.floor((limit * 2 + white) / (2 * white))
            assert px.ibits == bound.bit_length()


def test_getitem(pixels_subclass):
    # Check indexing into an array of rank zero.
    px = Pixels(42, fbits=0, limit=42)
    assert isinstance(px, pixels_subclass)
    assert px[...].numerators[()] == 42
    assert px[()].numerators[()] == 42
    # Check all sorts of 1D indexing schemes.
    px = Pixels([0, 1], fbits=0)
    assert px[...].numerators[0] == 0
    assert px[...].numerators[1] == 1
    assert px[:].numerators[0] == 0
    assert px[:].numerators[1] == 1
    assert px[:-1].numerators[0] == 0
    assert px[::-1].numerators[0] == 1
    assert px[::-1].numerators[1] == 0
    assert px[0:1].numerators[0] == 0
    assert px[1:2].numerators[0] == 1
    assert px[0].numerators[()] == 0
    assert px[1].numerators[()] == 1
    # Check indexing into an array of rank two.
    px = Pixels([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], fbits=0, limit=9)
    assert isinstance(px, pixels_subclass)
    for i in range(5):
        row = px[i]
        assert row.shape == (2,)
        for j in range(2):
            expected = (i * 2 + j)
            assert expected == row[j].to_array()[()]
            assert expected == px[i, j].to_array()[()]
    # Now do the full check of up to rank four.
    for shape in chain(*[permutations((0, 1, 2, 3), k) for k in range(4)]):
        rank = len(shape)
        size = math.prod(shape)
        data = list(range(size))
        array = SimpleArray(data, shape)
        px = Pixels(array, limit=max(0, size-1))
        assert isinstance(px, pixels_subclass)
        assert px[...].shape == shape
        # Index every individual element.
        for index in product(*[range(s) for s in shape]):
            assert array[index] == px[index].to_array()[()]
        # Slicing.
        slices = [slice(None), slice(1, None), slice(None, -1), slice(1, -1, 2)]
        for index in product(* [slices] * rank):
            selection = px[index]
            for slc, nold, nnew in zip(index, shape, selection.shape):
                assert len(range(*slc.indices(nold))) == nnew


def test_permute(pixels_subclass):
    # Rank zero.
    px = Pixels(42, fbits=0, limit=42)
    isinstance(px, pixels_subclass)
    assert px.permute().to_array()[()] == 42
    # Rank one.
    px = Pixels([0, 1], fbits=0)
    assert isinstance(px, pixels_subclass)
    for permutation in [(), (0,)]:
        assert px.permute(*permutation).to_array()[0] == 0
        assert px.permute(*permutation).to_array()[1] == 1
    # Rank two.
    data = [[1, 2], [3, 4]]
    px = Pixels(data, fbits=0, limit=4)
    assert isinstance(px, pixels_subclass)
    original = px.permute(0, 1)
    flipped = px.permute(1, 0)
    for i, j in product([0, 1], [0, 1]):
        assert original[i, j].to_array()[()] == data[i][j]
        assert flipped[i, j].to_array()[()] == data[j][i]
    # Arbitrary rank.
    for shape in chain(*[permutations((0, 1, 2, 3), k) for k in range(4)]):
        rank = len(shape)
        size = math.prod(shape)
        data = list(range(size))
        array = SimpleArray(data, shape)
        px = Pixels(array, limit=max(0, size-1))
        assert isinstance(px, pixels_subclass)
        assert px.permute().shape == shape
        for permutation in chain(*[permutations(range(k)) for k in range(rank)]):
            flipped1 = px.permute(*permutation).to_array()
            flipped2 = px.permute(permutation).to_array()
            for index in itertools.product(*[range(s) for s in shape]):
                other = tuple(index[p] for p in permutation) + index[len(permutation):]
                assert array[index] == flipped1[other]
                assert array[index] == flipped2[other]


def test_nroadcast_to(pixels_subclass):
    for shape in chain(*[permutations((0, 5, 7), k) for k in range(3)]):
        size = math.prod(shape)
        data = list(range(size))
        rank1 = len(shape)
        array1 = SimpleArray(data, shape)
        px1 = Pixels(array1, limit=max(0, size-1), fbits=0)
        assert isinstance(px1, pixels_subclass)
        for suffix in chain(*[permutations((0, 1, 2, 3), k) for k in range(4)]):
            px2 = px1.broadcast_to(shape + suffix)
            array2 = px2.to_array()
            for index in product(*[range(s) for s in px2.shape]):
                assert array2[index] == array1[index[:rank1]]


def test_bool(pixels_subclass):
    for true, false in zip((Pixels(1), Pixels([1, 1]), Pixels([[1, 1], [1, 1]])),
                           (Pixels(0), Pixels([0, 0]), Pixels([[0, 0], [0, 0]]))):
        assert isinstance(true, pixels_subclass)
        assert isinstance(false, pixels_subclass)
        # bool
        assert true
        # not
        assert (~ false)
        assert not (~ true)
        # and
        assert (true & true)
        assert not (true & false)
        assert not (false & true)
        assert not (false & false)
        # or
        assert (true | true)
        assert (true | false)
        assert (false | true)
        assert not (false | false)
        # xor
        assert not (true ^ true)
        assert (true ^ false)
        assert (false ^ true)
        assert not (false ^ false)
    # Ensure that Pixels are true if any element is non-zero.
    assert Pixels([[[1, 0]]])
    assert Pixels([[[0.5, 0]]], fbits=1)
    assert Pixels([[[0.5, 0]]], fbits=0)


def test_shifts(pixels_subclass):
    px = Pixels(1, fbits=0)
    assert isinstance(px, pixels_subclass)
    for shift in range(30):
        assert (px << shift).numerators[()] == 2**shift
        assert (px >> shift).numerators[()] == 1
        assert ((px >> shift) << shift).numerators[()] == 1
        assert ((px << shift) >> shift).numerators[()] == 2**shift


def irange(beg, end, length):
    step = (end - beg) / length
    return [round(beg + k*step) for k in range(length)]


def generate_pixels(shape) -> list[Pixels]:
    size = math.prod(shape)
    # generate a lot of test data.
    ps: list[Pixels] = []
    for fbits in (0, 1, 7, 8, 9):
        white = 2**fbits
        a1 = SimpleArray(irange(0, white+1, size), shape)
        a2 = SimpleArray(irange(white, -1, size), shape)
        a3 = SimpleArray(irange(0, white*3+1, size), shape)
        for a, limit in [(a1, white), (a2, white), (a3, 3*white)]:
            p = Pixels(a, white=white, limit=limit, fbits=fbits)
            ps.append(p)
            if len(shape) > 0:
                ps.append(p[::-1])
            if len(shape) > 1:
                ps.append(p[::-1, ::-1])
    return ps


two_arg_test = Callable[[Frac, Frac, Frac], bool]


def two_arg_map_values(a: Pixels, b: Pixels, r: Pixels, fn: two_arg_test):
    assert a.shape == b.shape == r.shape
    na, da = a.numerators, a.denominator
    nb, db = b.numerators, b.denominator
    nr, dr = r.numerators, r.denominator
    for index in product(*[range(s) for s in a.shape]):
        va = frac(na[*index], da)
        vb = frac(nb[*index], db)
        vr = frac(nr[*index], dr)
        assert fn(va, vb, vr)


def test_two_arg_fns(pixels_subclass):
    for shape in [(), (3,), (2, 3)]:
        ps = generate_pixels(shape)
        # __pos__
        for p in ps:
            pos = +p
            assert isinstance(pos, pixels_subclass)
            assert pos == p
        # __add__
        for a, b in product(ps, ps):
            r = a + b
            two_arg_map_values(a, b, r, lambda x, y, z: clip(x + y, 0, 1) == z)
        # __sub__
        for a, b in product(ps, ps):
            r = a - b
            two_arg_map_values(a, b, r, lambda x, y, z: clip(x - y, 0, 1) == z)
        # __mul__
        for a, b in product(ps, ps):
            r = a * b
            n = max(a.fbits, b.fbits)
            two_arg_map_values(a, b, r, lambda x, y, z: fracround(clip(x * y, 0, 1), n) == z)
        # __pow__
        # __truediv__
        # __floordiv__
        # __mod__
        # __lt__
        for a, b in product(ps, ps):
            r = a < b
            two_arg_map_values(a, b, r, lambda x, y, z: (1 if x < y else 0) == z)
        # __gt__
        for a, b in product(ps, ps):
            r = a > b
            two_arg_map_values(a, b, r, lambda x, y, z: (1 if x > y else 0) == z)
        # __le__
        for a, b in product(ps, ps):
            r = a <= b
            two_arg_map_values(a, b, r, lambda x, y, z: (1 if x <= y else 0) == z)
        # __ge__
        for a, b in product(ps, ps):
            r = a >= b
            two_arg_map_values(a, b, r, lambda x, y, z: (1 if x >= y else 0) == z)
        # __eq__
        for a, b in product(ps, ps):
            r = a == b
            two_arg_map_values(a, b, r, lambda x, y, z: (1 if x == y else 0) == z)
        # __ne__
        for a, b in product(ps, ps):
            r = a != b
            two_arg_map_values(a, b, r, lambda x, y, z: (1 if x != y else 0) == z)


def test_rolling_sum(pixels_subclass):
    px = Pixels([0.00, 0.25, 0.50, 0.75, 1.00], fbits=2)
    rs1 = px.rolling_sum(1)
    rs2 = px.rolling_sum(2)
    rs3 = px.rolling_sum(3)
    rs4 = px.rolling_sum(4)
    rs5 = px.rolling_sum(5)
    for rs in [rs1, rs2, rs3, rs4, rs5]:
        assert isinstance(rs, pixels_subclass)
    assert rs1 == px
    assert rs2 == Pixels([0.25, 0.75, 1.25, 1.75], limit=1.75, fbits=2)
    assert rs3 == Pixels([0.75, 1.50, 2.25], limit=2.25, fbits=2)
    assert rs4 == Pixels([1.50, 2.50], limit=2.5, fbits=2)
    assert rs5 == Pixels([2.5], limit=2.5, fbits=2)
