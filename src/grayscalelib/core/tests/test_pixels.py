from typing import Callable, Iterable
import pytest
import itertools
import math
from itertools import chain, permutations, product
from grayscalelib.core.pixels import Pixels, pixels_type
from grayscalelib.core.simplearray import SimpleArray


def clip(x, lo, hi):
    return max(lo, min(x, hi))


@pytest.fixture
def pixels_subclass(request):
    assert issubclass(request.param, Pixels)
    with pixels_type(request.param):
        yield request.param


def test_init(pixels_subclass):
    # Check some trivial cases.
    assert Pixels(0, power=0).to_array()[()] == 0
    assert Pixels(1, power=0).to_array()[()] == 1
    # round to nearest even
    assert Pixels(0.5, power=0).to_array()[()] == 0
    assert Pixels(1.5, limit=2, power=0).to_array()[()] == 2
    # Ensure that shapes are computed correctly.
    assert Pixels(0).shape == ()
    assert Pixels([]).shape == (0,)
    assert Pixels([[]]).shape == (1, 0)
    assert Pixels([[[]]]).shape == (1, 1, 0)
    assert Pixels([[0], [1]]).shape == (2, 1)
    # When converting real numbers in the [0, 1] range to pixels, the resulting
    # round-off error should be at most 2**(-power-1).
    for power in range(0, -14, -1):
        maxerr = 2**(-power-1)
        denominator = 2**(-power)
        size = denominator+1
        data = [n / denominator for n in range(size)]
        px = Pixels(data, power=power)
        assert isinstance(px, pixels_subclass)
        array = px.to_array()
        for index in range(size):
            assert abs(array[index] - data[index]) < maxerr
    # Check that black, white, and limit are correctly accounted for.
    for power, black in product([0, -1, -2], range(-3, 3)):
        for white, limit in product(range(black+1, black+3), range(black+1, black+9)):
            delta = (white - black)
            data = [[black-1, black, black+1],
                    [white-1, white, white+1],
                    [limit-1, limit, limit+1]]
            px = Pixels(data, power=power, black=black, white=white, limit=limit)
            pxdata = px.data
            for i, j in product(*tuple(range(s) for s in px.shape)):
                expected = max(black, min(data[i][j], limit))
                restored = ((pxdata[i, j] * delta) * 2**power) + black
                assert black <= restored <= limit + delta / 2
                assert abs(restored - expected) <= delta


def test_getitem(pixels_subclass):
    # Check indexing into an array of rank zero.
    px = Pixels(42, power=0, limit=42)
    assert isinstance(px, pixels_subclass)
    assert px[...].data[()] == 42
    assert px[()].data[()] == 42
    # Check all sorts of 1D indexing schemes.
    px = Pixels([0, 1], power=0)
    assert px[...].data[0] == 0
    assert px[...].data[1] == 1
    assert px[:].data[0] == 0
    assert px[:].data[1] == 1
    assert px[:-1].data[0] == 0
    assert px[::-1].data[0] == 1
    assert px[::-1].data[1] == 0
    assert px[0:1].data[0] == 0
    assert px[1:2].data[0] == 1
    assert px[0].data[()] == 0
    assert px[1].data[()] == 1
    # Check indexing into an array of rank two.
    px = Pixels([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], power=0, limit=9)
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
    px = Pixels(42, power=0, limit=42)
    isinstance(px, pixels_subclass)
    assert px.permute().to_array()[()] == 42
    # Rank one.
    px = Pixels([0, 1], power=0)
    assert isinstance(px, pixels_subclass)
    for permutation in [(), (0,)]:
        assert px.permute(*permutation).to_array()[0] == 0
        assert px.permute(*permutation).to_array()[1] == 1
    # Rank two.
    data = [[1, 2], [3, 4]]
    px = Pixels(data, power=0, limit=4)
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
        px1 = Pixels(array1, limit=max(0, size-1), power=0)
        assert isinstance(px1, pixels_subclass)
        for suffix in chain(*[permutations((0, 1, 2, 3), k) for k in range(4)]):
            px2 = px1.broadcast_to(shape + suffix)
            array2 = px2.to_array()
            for index in product(*[range(s) for s in px2.shape]):
                assert array2[index] == array1[index[:rank1]]


def test_bool(pixels_subclass):
    for alltrue, anytrue, allfalse in zip((Pixels(1), Pixels([1, 1]), Pixels([[1, 1], [1, 1]])),
                                          (Pixels(1), Pixels([0, 1]), Pixels([[0, 0], [1, 0]])),
                                          (Pixels(0), Pixels([0, 0]), Pixels([[0, 0], [0, 0]]))):
        assert isinstance(alltrue, pixels_subclass)
        assert isinstance(anytrue, pixels_subclass)
        assert isinstance(allfalse, pixels_subclass)
        # bool
        assert alltrue.all()
        assert alltrue.any()
        assert anytrue.any()
        # not
        assert (~ allfalse).all()
        assert not (~ alltrue).all()
        assert not (~ alltrue).any()
        assert not (~ anytrue).all()
        # and
        assert (alltrue & alltrue).all()
        assert (alltrue & alltrue).any()
        assert (alltrue & anytrue).any()
        assert not (alltrue & allfalse).any()
        assert not (allfalse & alltrue).any()
        assert not (allfalse & allfalse).any()
        # or
        assert (alltrue | alltrue).all()
        assert (alltrue | alltrue).any()
        assert (alltrue | allfalse).all()
        assert (alltrue | allfalse).any()
        assert (allfalse | alltrue).all()
        assert (allfalse | alltrue).any()
        assert not (allfalse | allfalse).all()
        assert not (allfalse | allfalse).any()
        # xor
        assert not (alltrue ^ alltrue).any()
        assert (alltrue ^ allfalse).all()
        assert (allfalse ^ alltrue).all()
        assert not (allfalse ^ allfalse).any()
    assert Pixels([[[1, 1]]]).all()
    assert Pixels([[[0.5, 1.0]]], power=-1).all()
    assert not Pixels([[[0.5, 0]]], power=0).all()
    assert not Pixels([[[0.5, 0]]], power=0).all()
    assert not Pixels([[[0.5, 0]]], power=0).any()


def test_shifts(pixels_subclass):
    px = Pixels(1, power=0)
    assert isinstance(px, pixels_subclass)
    for shift in range(30):
        assert (px << shift).data[()] == 1
        assert (px >> shift).data[()] == 1
        assert ((px >> shift) << shift).data[()] == 1
        assert ((px << shift) >> shift).data[()] == 1


def irange(beg, end, length):
    step = (end - beg) / length
    return [round(beg + k*step) for k in range(length)]


def generate_pixels(shape) -> list[Pixels]:
    size = math.prod(shape)
    # generate a lot of test data.
    ps: list[Pixels] = []
    for power in (0, -1, -7, -8, -9):
        white = 2**(-power)
        a1 = SimpleArray(irange(0, white+1, size), shape)
        a2 = SimpleArray(irange(white, -1, size), shape)
        a3 = SimpleArray(irange(0, white*3+1, size), shape)
        for a, limit in [(a1, white), (a2, white), (a3, 3*white)]:
            p = Pixels(a, white=white, limit=limit, power=power)
            ps.append(p)
            if len(shape) > 0:
                ps.append(p[::-1])
            if len(shape) > 1:
                ps.append(p[::-1, ::-1])
    return ps


def test_pow(pixels_subclass):
    a = Pixels(0.5, power=-2)
    b = Pixels(1.0, power=-2)
    assert isinstance(a, pixels_subclass)
    assert isinstance(b, pixels_subclass)
    assert (a ** 1).data[()] == 2
    assert (a ** 2).data[()] == 1
    assert (a ** 3).data[()] == 0
    assert (a ** 0.5).data[()] == 3
    assert (b ** 1).data[()] == 4
    assert (b ** 2).data[()] == 4
    assert (b ** 3).data[()] == 4
    assert (b ** 0.5).data[()] == 4


two_arg_test = Callable[[float, float, float], bool]


def pixel_values(*pixels: Pixels) -> Iterable[tuple[float, ...]]:
    if len(pixels) == 0:
        return
    shape = pixels[0].shape
    for px in pixels[1:]:
        assert px.shape == shape
    arrays = [px.data for px in pixels]
    scales: list[float] = [px.scale for px in pixels]
    for index in product(*[range(s) for s in shape]):
        yield tuple(a[index] * s for a, s in zip(arrays, scales))


def test_two_arg_fns(pixels_subclass):
    for shape in [(), (3,), (2, 3)]:
        ps = generate_pixels(shape)
        # __pos__
        for p in ps:
            pos = +p
            assert isinstance(pos, pixels_subclass)
            assert (pos == p).all()
        # __add__
        for a, b in product(ps, ps):
            for x, y, z in pixel_values(a, b, a + b):
                assert clip(x + y, 0, 1) == z
        # __sub__
        for a, b in product(ps, ps):
            for x, y, z in pixel_values(a, b, a - b):
                assert clip(x - y, 0, 1) == z
        # __mul__
        for a, b in product(ps, ps):
            r = a * b
            for x, y, z in pixel_values(a, b, r):
                assert abs(clip(x * y, 0, 1) - z) <= (r.scale / 2)
        # __truediv__
        for a, b in product(ps, ps):
            r = a / b
            for x, y, z in pixel_values(a, b, r):
                if y == 0:
                    assert z == 1
                else:
                    assert abs(clip(x / y, 0, 1) - z) <= (r.scale / 2)
        # __mod__
        for a, b in product(ps, ps):
            r = a % b
            for x, y, z in pixel_values(a, b, r):
                if y == 0:
                    assert z == 0
                else:
                    assert abs((x % y) - z) <= (r.scale / 2)
        # __lt__
        for a, b in product(ps, ps):
            for x, y, z in pixel_values(a, b, a < b):
                assert z == 1 if x < y else z == 0
        # __gt__
        for a, b in product(ps, ps):
            for x, y, z in pixel_values(a, b, a > b):
                assert z == 1 if x > y else z == 0
        # __le__
        for a, b in product(ps, ps):
            for x, y, z in pixel_values(a, b, a <= b):
                assert z == 1 if x <= y else z == 0
        # __ge__
        for a, b in product(ps, ps):
            for x, y, z in pixel_values(a, b, a >= b):
                assert z == 1 if x >= y else z == 0
        # __eq__
        for a, b in product(ps, ps):
            for x, y, z in pixel_values(a, b, a == b):
                assert z == 1 if x == y else z == 0
        # __ne__
        for a, b in product(ps, ps):
            for x, y, z in pixel_values(a, b, a != b):
                assert z == 1 if x != y else z == 0


def test_rolling_sum(pixels_subclass):
    # 0D tests
    assert (Pixels(0).rolling_sum(()) == Pixels(0)).all()
    assert (Pixels(1).rolling_sum(()) == Pixels(1)).all()
    # 1D tests
    px = Pixels([0.00, 0.25, 0.50, 0.75, 1.00], power=-2)
    rs1 = px.rolling_sum(1)
    rs2 = px.rolling_sum(2)
    rs3 = px.rolling_sum(3)
    rs4 = px.rolling_sum(4)
    rs5 = px.rolling_sum(5)
    for rs in [rs1, rs2, rs3, rs4, rs5]:
        assert isinstance(rs, pixels_subclass)
    assert (rs1 == px).all()
    assert (rs2 == Pixels([0.25, 0.75, 1.25, 1.75], limit=1.75, power=-2)).all()
    assert (rs3 == Pixels([0.75, 1.50, 2.25], limit=2.25, power=-2)).all()
    assert (rs4 == Pixels([1.50, 2.50], limit=2.5, power=-2)).all()
    assert (rs5 == Pixels([2.5], limit=2.5, power=-2)).all()
    # 2D tests
    px = Pixels([[0, 1, 2], [3, 4, 5], [6, 7, 8]], limit=8, power=0)
    assert (px.rolling_sum((1, 1)) == Pixels([[0, 1, 2], [3, 4, 5], [6, 7, 8]], limit=8, power=0)).all()
    assert (px.rolling_sum((2, 1)) == Pixels([[3, 5, 7], [9, 11, 13]], limit=13, power=0)).all()
    assert (px.rolling_sum((3, 1)) == Pixels([[9, 12, 15]], limit=15, power=0)).all()
    assert (px.rolling_sum((1, 2)) == Pixels([[1, 3], [7, 9], [13, 15]], limit=15, power=0)).all()
    assert (px.rolling_sum((2, 2)) == Pixels([[8, 12], [20, 24]], limit=24, power=0)).all()
    assert (px.rolling_sum((3, 2)) == Pixels([[21, 27]], limit=27, power=0)).all()
    assert (px.rolling_sum((1, 3)) == Pixels([[3], [12], [21]], limit=21, power=0)).all()
    assert (px.rolling_sum((2, 3)) == Pixels([[15], [33]], limit=33, power=0)).all()
    assert (px.rolling_sum((3, 3)) == Pixels([[36]], limit=36, power=0)).all()
    assert (px.rolling_sum(1) == px.rolling_sum((1,))).all()
    assert (px.rolling_sum(1) == px.rolling_sum((1, 1))).all()
    assert (px.rolling_sum(2) == px.rolling_sum((2,))).all()
    assert (px.rolling_sum(2) == px.rolling_sum((2, 1))).all()
    assert (px.rolling_sum(3) == px.rolling_sum((3,))).all()
    assert (px.rolling_sum(3) == px.rolling_sum((3, 1))).all()
