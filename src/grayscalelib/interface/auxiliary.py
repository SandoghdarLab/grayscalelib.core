from __future__ import annotations


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


def permute_check_before(permutation: tuple[int, ...], rank: int):
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


def permute_check_after(permutation: tuple[int, ...], oldshape: tuple, newshape: tuple):
    # These checks should always pass unless there is a bug in an
    # implementation of _permute_, so there is no need for elaborate error
    # messages.
    assert len(permutation) == len(oldshape) == len(newshape)
    for i, p in enumerate(permutation):
        assert newshape[i] == oldshape[p]
