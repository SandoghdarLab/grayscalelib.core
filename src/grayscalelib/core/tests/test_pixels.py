import pytest
import itertools
import math
import sys
import numpy as np
import numpy.typing as npt
from itertools import chain, permutations, product
from grayscalelib.core.discretization import Discretization
from grayscalelib.core.pixels import Pixels, pixels_type, broadcast


@pytest.fixture
def pixels_subclass(request):
    cls = request.param
    assert issubclass(cls, Pixels)
    with pixels_type(cls):
        yield cls


def generate_pixels(shape: tuple[int, ...]) -> list[Pixels]:
    """Return a list of interesting Pixels objects of the supplied shape."""
    size = math.prod(shape)
    pixels: list[Pixels] = []
    for states in (8, 256):
        for black in (-2, -1, 0, 1, 2):
            for white in (-2, -1, 0, 1, 2):
                lo = min(black, white)
                hi = max(black, white)
                px = Pixels(np.linspace(lo, hi, size), black=black, white=white, states=states)
                assert np.all(lo <= px.data)
                assert np.all(px.data <= hi)
                pixels.append(px)
    return pixels


def test_pixels_init(pixels_subclass):
    # Black as the only state.
    px = Pixels(0, states=1, white=0.0)
    assert px.raw[()] == 0
    assert px.states == 1
    assert px.discretization.eps == 0.0
    assert px.discretization == Discretization((0.0, 0.0), (0, 0))

    # White as the only state.
    px = Pixels(0, states=1, black=1.0)
    assert px.raw[()] == 0
    assert px.states == 1
    assert px.discretization.eps == 0.0
    assert px.discretization == Discretization((1.0, 1.0), (0, 0))

    # Boolean pixels.
    px = Pixels([1, 0], states=2)
    assert px.raw[0] == 1
    assert px.raw[1] == 0
    assert px.states == 2
    assert px.discretization.eps == 1.0
    assert px.discretization == Discretization((0.0, 1.0), (0, 1))

    # Round to nearest even.
    assert Pixels(0.5, states=2).raw[()] == 0
    assert Pixels(0.5, states=4).raw[()] == 2

    # Ensure that shapes are computed correctly.
    assert Pixels(0).shape == ()
    assert Pixels([]).shape == (0,)
    assert Pixels([[]]).shape == (1, 0)
    assert Pixels([[[]]]).shape == (1, 1, 0)
    assert Pixels([[0], [1]]).shape == (2, 1)

    # Ensure proper discretization.
    for states in range(2, 257, 1):
        # Multiples of eps should have an exact representation.
        d = (states - 1)
        px = Pixels([n / d for n in range(d + 1)], states=states)
        assert px.eps == 1 / d
        assert isinstance(px, pixels_subclass)
        for index in range(d + 1):
            assert px.raw[index] == index
        # Rounding should always be less than or equal to eps/2.
        d = 2 * (states - 1)
        values = [n / d for n in range(d + 1)]
        px = Pixels(values, states=states)
        results = px.data
        for index in range(d + 1):
            assert abs(results[index] - values[index]) <= px.roundoff


def test_pixels_getitem(pixels_subclass):
    # Check indexing into an array of rank zero.
    px = Pixels(42, states=1, black=42, white=42)
    assert isinstance(px, pixels_subclass)
    assert px[...].data[()] == 42
    assert px[()].data[()] == 42

    # Check all sorts of 1D indexing schemes.
    px = Pixels([0, 1], states=2)
    assert px[...].raw[0] == 0
    assert px[...].raw[1] == 1
    assert px[:].raw[0] == 0
    assert px[:].raw[1] == 1
    assert px[:-1].raw[0] == 0
    assert px[::-1].raw[0] == 1
    assert px[::-1].raw[1] == 0
    assert px[0:1].raw[0] == 0
    assert px[1:2].raw[0] == 1
    assert px[0].raw[()] == 0
    assert px[1].raw[()] == 1

    # Check indexing into an array of rank two.
    px = Pixels([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], white=9, states=10)
    assert isinstance(px, pixels_subclass)
    for i in range(5):
        row = px[i]
        assert row.shape == (2,)
        for j in range(2):
            expected = (i * 2 + j)
            assert expected == row[j].raw[()]
            assert expected == px[i, j].raw[()]

    # Now do the full check of up to rank four.
    for shape in chain(*[permutations((0, 1, 2, 3), k) for k in range(4)]):
        rank = len(shape)
        size = math.prod(shape)
        values = np.reshape(list(range(size)), shape)
        px = Pixels(values, white=size-1, states=size)
        assert isinstance(px, pixels_subclass)
        assert px[...].shape == shape
        # Index every individual element.
        for index in product(*[range(s) for s in shape]):
            assert values[index] == px[index].data[()]
        # Slicing.
        slices = [slice(None), slice(1, None), slice(None, -1), slice(1, -1, 2)]
        for index in product(* [slices] * rank):
            selection = px[index]
            for slc, nold, nnew in zip(index, shape, selection.shape):
                assert len(range(*slc.indices(nold))) == nnew


def test_pixels_permute(pixels_subclass):
    # Rank zero.
    px = Pixels(42, states=1, white=42, black=42)
    isinstance(px, pixels_subclass)
    assert px.permute().data[()] == 42

    # Rank one.
    px = Pixels([0, 1], states=2)
    assert isinstance(px, pixels_subclass)
    for permutation in [(), (0,)]:
        assert px.permute(*permutation).raw[0] == 0
        assert px.permute(*permutation).raw[1] == 1

    # Rank two.
    values = [[1, 2], [3, 4]]
    px = Pixels(values, black=1, white=4, states=4)
    assert isinstance(px, pixels_subclass)
    original = px.permute(0, 1)
    flipped = px.permute(1, 0)
    for i, j in product([0, 1], [0, 1]):
        assert px[i, j].data[()] == values[i][j]
        assert original[i, j].data[()] == values[i][j]
        assert flipped[i, j].data[()] == values[j][i]

    # Arbitrary rank.
    for shape in chain(*[permutations((0, 1, 2, 3), k) for k in range(4)]):
        rank = len(shape)
        size = math.prod(shape)
        values = np.reshape(list(range(size)), shape)
        px = Pixels(values, white=size-1, states=size)
        assert isinstance(px, pixels_subclass)
        assert px.permute().shape == shape
        for permutation in chain(*[permutations(range(k)) for k in range(rank)]):
            flipped1 = px.permute(*permutation).raw
            flipped2 = px.permute(permutation).raw
            for index in itertools.product(*[range(s) for s in shape]):
                other = tuple(index[p] for p in permutation) + index[len(permutation):]
                assert values[index] == flipped1[other]
                assert values[index] == flipped2[other]


def test_pixels_reencode(pixels_subclass):
    # Changes in the number of states.
    px = Pixels([0, 1], states=2)
    for n in range(1, 5):
        assert isinstance(px, pixels_subclass)
        raw = px.reencode(states=n+1).raw
        assert raw[0] == 0
        assert raw[1] == n

    # Changes of black and white.
    px = Pixels([0, 1], states=2)
    assert px.reencode(black=0).states == 2
    assert px.reencode(white=1).states == 2
    assert px.reencode(white=2).raw[1] == 0
    assert px.reencode(black=-1).raw[0] == 0

    # Changes of eps.
    px = Pixels([0, 0.25, 0.5, 0.75, 1], states=256)
    raw = px.reencode(states=5).raw
    assert np.all(raw == [0, 1, 2, 3, 4])

    # Stress test.
    for shape in [(3,), (9,), (24,)]:
        for px in generate_pixels(shape):
            for states in (1, 2, 8, 255, 256, 257):
                for db in (0, 1, 2):
                    for dw in (0, 1, 2):
                        black = px.black - db
                        white = px.white + dw
                        new = px.reencode(black=black, white=white, states=states)
                        before = px.data
                        after = px.data
                        assert np.allclose(before, after, rtol=0, atol=new.roundoff)


def test_pixels_reshape(pixels_subclass):
    pass # TODO


def test_pixels_broadcast(pixels_subclass):
    # Test adding of additional trailing axes.
    for shape in chain(*[permutations((0, 5, 7), k) for k in range(3)]):
        size = math.prod(shape)
        rank1 = len(shape)
        array1 = np.reshape(list(range(size)), shape)
        px1 = Pixels(array1, white=size-1, states=size)
        assert isinstance(px1, pixels_subclass)
        for suffix in chain(*[permutations((0, 1, 2, 3), k) for k in range(4)]):
            px2 = px1.broadcast_to(shape + suffix)
            array2 = px2.raw
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
    assert Pixels([[[0.5, 1.0]]], states=3).all()
    assert not Pixels([[[0.5, 0]]], states=2).all()
    assert not Pixels([[[0.5, 0]]], states=2).all()
    assert not Pixels([[[0.5, 0]]], states=2).any()


# def test_shifts(pixels_subclass):
#     px = Pixels(1)
#     assert isinstance(px, pixels_subclass)
#     for shift in range(30):
#         assert (px << shift).raw[()] == 1
#         assert (px >> shift).raw[()] == 1
#         assert ((px >> shift) << shift).raw[()] == 1
#         assert ((px << shift) >> shift).raw[()] == 1


def test_two_arg_fns(pixels_subclass):
    def err(a: Pixels, b: Pixels,):
        eps = 0.0
        if a.states > 1:
            eps = min(eps, (a.white - a.black) / (a.states - 1))
        if b.states > 1:
            eps = min(eps, (b.white - b.black) / (b.states - 1))
        return (eps + sys.float_info.epsilon) / 2


    for shape in [(), (3,), (3, 2)]:
        pixels = generate_pixels(shape)
        pairs = [broadcast(a, b) for (a, b) in product(pixels, pixels)]
        # __pos__
        for p in pixels:
            pos = +p
            assert isinstance(pos, pixels_subclass)
            assert np.all(pos.data == p.data)
        # __add__
        for a, b in pairs:
            result = (a + b)
            assert np.allclose(a.data+b.data, result.data, rtol=0, atol=result.roundoff)
        # __sub__
        for a, b in pairs:
            result = (a - b)
            assert np.allclose(a.data-b.data, result.data, rtol=0, atol=result.roundoff)
        # __mul__
        for a, b in pairs:
            result = (a * b)
            assert np.allclose(a.data*b.data, result.data, rtol=0, atol=result.roundoff)


# def test_rolling_sum(pixels_subclass):
#     # 0D tests
#     assert (Pixels(0).rolling_sum(()) == Pixels(0)).all()
#     assert (Pixels(1).rolling_sum(()) == Pixels(1)).all()
#     # 1D tests
#     px = Pixels([0.00, 0.25, 0.50, 0.75, 1.00], power=-2)
#     rs1 = px.rolling_sum(1)
#     rs2 = px.rolling_sum(2)
#     rs3 = px.rolling_sum(3)
#     rs4 = px.rolling_sum(4)
#     rs5 = px.rolling_sum(5)
#     for rs in [rs1, rs2, rs3, rs4, rs5]:
#         assert isinstance(rs, pixels_subclass)
#     assert (rs1 == px).all()
#     assert (rs2 == Pixels([0.25, 0.75, 1.25, 1.75], limit=1.75, power=-2)).all()
#     assert (rs3 == Pixels([0.75, 1.50, 2.25], limit=2.25, power=-2)).all()
#     assert (rs4 == Pixels([1.50, 2.50], limit=2.5, power=-2)).all()
#     assert (rs5 == Pixels([2.5], limit=2.5, power=-2)).all()
#     # 2D tests
#     px = Pixels([[0, 1, 2], [3, 4, 5], [6, 7, 8]], limit=8, power=0)
#     assert (px.rolling_sum((1, 1)) == Pixels([[0, 1, 2], [3, 4, 5], [6, 7, 8]], limit=8, power=0)).all()
#     assert (px.rolling_sum((2, 1)) == Pixels([[3, 5, 7], [9, 11, 13]], limit=13, power=0)).all()
#     assert (px.rolling_sum((3, 1)) == Pixels([[9, 12, 15]], limit=15, power=0)).all()
#     assert (px.rolling_sum((1, 2)) == Pixels([[1, 3], [7, 9], [13, 15]], limit=15, power=0)).all()
#     assert (px.rolling_sum((2, 2)) == Pixels([[8, 12], [20, 24]], limit=24, power=0)).all()
#     assert (px.rolling_sum((3, 2)) == Pixels([[21, 27]], limit=27, power=0)).all()
#     assert (px.rolling_sum((1, 3)) == Pixels([[3], [12], [21]], limit=21, power=0)).all()
#     assert (px.rolling_sum((2, 3)) == Pixels([[15], [33]], limit=33, power=0)).all()
#     assert (px.rolling_sum((3, 3)) == Pixels([[36]], limit=36, power=0)).all()
#     assert (px.rolling_sum(1) == px.rolling_sum((1,))).all()
#     assert (px.rolling_sum(1) == px.rolling_sum((1, 1))).all()
#     assert (px.rolling_sum(2) == px.rolling_sum((2,))).all()
#     assert (px.rolling_sum(2) == px.rolling_sum((2, 1))).all()
#     assert (px.rolling_sum(3) == px.rolling_sum((3,))).all()
#     assert (px.rolling_sum(3) == px.rolling_sum((3, 1))).all()
