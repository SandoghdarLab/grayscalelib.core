from grayscalelib.core.discretization import Discretization


def test_discretization():
    # Corner cases.
    d = Discretization((1.0, 1.0), (0, 0))
    assert d.a == 0.0
    assert d.b == 0.0

    # Test various non-empty intervals.
    tuples = [(0, 1), (0, 3), (1, 3), (-3, 0), (0, 255), (0, 256)]
    codomains = tuples + [(i2, i1) for (i1, i2) in tuples]
    domains = [(float(i1), float(i2)) for (i1, i2) in codomains]
    for i1, i2 in codomains:
        for f1, f2 in domains:
            flip = (f2 < f1) ^ (i2 < i1)
            # Forward
            d = Discretization((f1, f2), (i1, i2))
            assert d.flip == flip
            assert d.domain.lo == min(f1, f2)
            assert d.domain.hi == max(f1, f2)
            assert d.codomain.lo == min(i1, i2)
            assert d.codomain.hi == max(i1, i2)
            assert (d(f1) == i1) ^ (d(f1) == i2)
            assert (d(f2) == i1) ^ (d(f2) == i2)
            assert d.states == abs(i2 - i1) + 1
            assert d.eps * (d.states - 1) == abs(f2 - f1)
            # Backward
            i = d.inverse
            assert i.domain.lo == min(i1, i2)
            assert i.domain.hi == max(i1, i2)
            assert i.codomain.lo == min(f1, f2)
            assert i.codomain.hi == max(f1, f2)
            assert (i(i1) == f1) ^ (i(i1) == f2)
            assert (i(i2) == f1) ^ (i(i2) == f2)
            # Roundtrip
            for y in range(d.codomain.lo, d.codomain.hi):
                assert d(i(y)) == y
