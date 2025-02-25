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
    for (ilo, ihi) in codomains:
        for (flo, fhi) in domains:
            flip = (fhi < flo) ^ (ihi < ilo)
            # Forward
            d = Discretization((flo, fhi), (ilo, ihi))
            assert d.domain.lo == min(flo, fhi)
            assert d.domain.hi == max(flo, fhi)
            assert d.codomain.lo == min(ilo, ihi)
            assert d.codomain.hi == max(ilo, ihi)
            assert d(flo) == (ihi if flip else ilo)
            assert d(fhi) == (ilo if flip else ihi)
            assert d.states == abs(ihi - ilo) + 1
            assert d.eps * (d.states - 1) == abs(fhi - flo)
            # Backward
            i = d.inverse
            assert i.domain.lo == min(ilo, ihi)
            assert i.domain.hi == max(ilo, ihi)
            assert i.codomain.lo == min(flo, fhi)
            assert i.codomain.hi == max(flo, fhi)
            assert i(ilo) == (fhi if flip else flo)
            assert i(ihi) == (flo if flip else fhi)
            # Roundtrip
            for y in range(d.codomain.lo, d.codomain.hi):
                assert d(i(y)) == y
