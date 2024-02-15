import pytest
import itertools
import math
from itertools import chain, permutations, product
from grayscalelib.core.pixels import Pixels, pixels_type
from grayscalelib.core.simplearray import SimpleArray


@pytest.fixture
def pixels_subclass(request):
    assert issubclass(request.param, Pixels)
    with pixels_type(request.param):
        yield request.param


def test_init(pixels_subclass):
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
            delta = (white - black)
            data = [[black-1, black, black+1],
                    [white-1, white, white+1],
                    [limit-1, limit, limit+1]]
            px = Pixels(data, fbits=fbits, black=black, white=white, limit=limit)
            numerators = px.numerators
            for i, j in product(*tuple(range(s) for s in px.shape)):
                expected = black if delta == 0 else max(black, min(data[i][j], limit))
                restored = ((numerators[i, j] * delta) >> fbits) + black
                assert black <= restored <= limit
                assert abs(restored - expected) <= delta
    # Ensure that the ibits are computed correctly.
    for white in range(1, 5):
        for limit in range(white, 17):
            data = list(range(-1, limit+1))
            px = Pixels(data, fbits=0, white=white, limit=limit)
            assert px.ibits == (limit // white).bit_length()


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
    sizes = (0, 1, 2, 3)
    for shape in chain(*[permutations(sizes, k) for k in range(4)]):
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
    sizes = (0, 1, 2, 3)
    for shape in chain(*[permutations(sizes, k) for k in range(4)]):
        rank = len(shape)
        size = math.prod(shape)
        data = list(range(size))
        array = SimpleArray(data, shape)
        px = Pixels(array, limit=max(0, size-1))
        assert isinstance(px, pixels_subclass)
        assert px.permute().shape == shape
        for permutation in chain(*[permutations(range(k)) for k in range(rank)]):
            flipped = px.permute(*permutation).to_array()
            for index in itertools.product(*[range(s) for s in shape]):
                other = tuple(index[p] for p in permutation) + index[len(permutation):]
                assert array[index] == flipped[other]
