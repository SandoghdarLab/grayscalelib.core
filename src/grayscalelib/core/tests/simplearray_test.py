from itertools import product, permutations
from math import prod
from grayscalelib.core.simplearray import SimpleArray

arrays: list[SimpleArray] = [
    SimpleArray([]),
    SimpleArray([], (0,)),
    SimpleArray([], (0, 42)),
    SimpleArray([], (42, 0)),
    SimpleArray([42], ()),
    SimpleArray([42], (1,)),
    SimpleArray([42], (1, 1)),
    SimpleArray([1, 2, 3, 4], (4,)),
    SimpleArray([1, 2, 3, 4], (2, 2)),
    SimpleArray([1, 2, 3, 4], (1, 4)),
    SimpleArray([1, 2, 3, 4], (4, 1)),
    SimpleArray([1, 2, 3, 4, 5, 6], (1, 2, 3)),
    SimpleArray([1, 2, 3, 4, 5, 6], (3, 2, 1)),
]

def test_simplearray():
    for array in arrays:
        shape = array.shape
        strides = array.strides
        values = array._values
        # Check the internal structure of each array.
        rank = len(shape)
        assert len(strides) == rank
        if rank > 0:
            assert strides[-1] == 1
        assert len(values) == prod(shape)
        # Check each array's __getitem__ method.
        for index in product(*tuple(range(s) for s in shape)):
            pos = sum(i * s for i, s in zip(index, strides))
            assert values[pos] == array[*index]
        # Check each array's permute method.
        for n in range(len(shape)):
            axes = tuple(range(n))
            for permutation in permutations(axes):
                parray = array.permute(permutation)
                for index in product(*tuple(range(s) for s in shape)):
                    pindex = tuple(index[p] for p in permutation) + index[n:]
                    assert array[index] == parray[pindex]
