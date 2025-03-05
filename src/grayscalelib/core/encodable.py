from typing import Any, Callable, Iterable, Type, TypeVar


def encoding_priority(cls: Type) -> int:
    if not hasattr(cls, "__encoding_priority__"):
        return -(2**31)
    return cls.__encoding_priority__()


def choose_encoding(first: type, *rest: type) -> type:
    """Among all the supplied encodings, choose the one with the highest priority."""
    if len(rest) == 0:
        return first
    else:
        return max(first, *rest, key=lambda cls: encoding_priority(cls))


# A database of direct conversion functions.
direct_encoders: dict[Type, dict[Type, Callable[[Any], Any]]] = {}


# A cache of all conversion functions created so far.  Invalidated whenever new
# direct encoders are registered
encoder_cache: dict[tuple[Type, Type], Callable[[Any], Any]] = {}


A = TypeVar("A")
B = TypeVar("B")


def register_direct_encoder(src: Type[A], dst: Type[B], encoder: Callable[[A], B]) -> None:
    encoder_cache.clear()
    direct_encoders.setdefault(src, {}).setdefault(dst, encoder)  # type: ignore


def shortest_encoder_path(src: Type, dst: Type) -> list[Type]:
    """
    Compute a list of encodings whose first element is the supplied source
    encoding, whose last element is the supplied destination encoding, and
    where any two consecutive elements have a direct encoder from the former to
    the latter.
    """

    # Find all encodings that can be reached in a single step.
    def neighbors(node: Type) -> Iterable[Type]:
        """"""
        if node in direct_encoders:
            return direct_encoders[node].keys()
        else:
            return []

    # Find all encodings that can be reached in a single step and that haven't
    # been visited before.
    visited: set[Type] = set()

    def unvisited_neighbors(node: Type) -> Iterable[Type]:
        targets = neighbors(node)
        for target in targets:
            if target not in visited:
                visited.add(target)
                yield target

    # Use Dijkstra's algorithm for finding the shortest path from one encoder
    # to the next one.
    paths: list[list[Type]] = [[src]]
    while True:
        # If we are at a node with an edge to the destination, that edge is the
        # last step of the shortest path.
        for path in paths:
            if dst in neighbors(path[-1]):
                return path + [dst]
        # If there is no edge to the destination, take all paths of length N
        # and construct all paths of length N+1 that don't contain circles.
        paths = [path + [cls] for path in paths for cls in unvisited_neighbors(path[-1])]
        # Give up if there are zero paths left.
        if len(paths) == 0:
            raise RuntimeError(f"Don't know how to encode {src} objects as {dst}.")


def find_encoder(cls1: Type[A], cls2: Type[B]) -> Callable[[A], B]:
    """Return an encoder from the former class to the latter."""
    # Check the encoder cache first.
    key = (cls1, cls2)
    if key in encoder_cache:
        return encoder_cache[key]

    # If there is no cached encoder, compute one and place it in the cache.
    path = shortest_encoder_path(cls1, cls2)

    def encoder(e1: A) -> B:
        encodable = e1
        for index in range(len(path) - 1):
            src = path[index]
            dst = path[index + 1]
            direct_encoder = direct_encoders[src][dst]
            encodable = direct_encoder(encodable)
        assert isinstance(encodable, cls2)
        return encodable

    encoder_cache[key] = encoder
    return encoder


def encode_as(obj: Any, cls: Type[A]) -> A:
    """Represent an object with a particular encoding."""
    if isinstance(obj, cls):
        return obj
    result = find_encoder(type(obj), cls)(obj)
    assert isinstance(result, cls)
    return result


class Encodable:
    """
    An object that can be encoded in one of several different ways.

    This base class is for objects that have one external interface, and
    multiple equivalent representations that can be chosen from.  Whenever an
    operation handles more than one Encodable at the same time, it can decide
    to first convert all of them into the same representation by means of the
    functions ``choose_encoding`` and ``encode_as``.
    """

    @classmethod
    def __encoding_priority__(cls) -> int:
        """ "How preferential to treat this class when choosing an encoding."""
        # Return minus one by default, so that any class with non-negative
        # priority takes precedence.
        return -1

    def encode_as(self, cls: Type[A]) -> A:
        """Represent this object with a particular encoding."""
        return encode_as(self, cls)
