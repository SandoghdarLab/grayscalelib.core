from __future__ import annotations

from typing import Protocol, NamedTuple

class ContinuousInterval(NamedTuple):
    """The closed interval of floating-point numbers."""
    lo: float
    hi: float


class DiscreteInterval(NamedTuple):
    """A closed interval of integers."""
    lo: int
    hi: int

class Discretization():
    """A function of the form f(x) = round(a * max(lo, min(x, hi)) + b)."""
    _domain: ContinuousInterval
    _codomain: DiscreteInterval
    _a: float
    _b: float
    _inverse: InverseDiscretization

    def __init__(
            self,
            domain: tuple[float, float],
            codomain: tuple[int, int],
            *,
            _inverse: InverseDiscretization | None = None):
        # Normalize the domain.
        if domain[1] < domain[0]:
            fhi, flo = domain
            i2, i1 = codomain
        else:
            (flo, fhi) = domain
            (i1, i2) = codomain
        # Compute the coefficients.
        dy = i2 - i1
        dx = fhi - flo
        a = dy / dx if dx > 0.0 else 0.0
        # Normalize the codomain.
        if i2 < i1:
            ilo, ihi = i2, i1
            a = -a
        else:
            ilo, ihi = i1, i2
        b = ilo - a * flo
        # Recompute ilo and ihi
        ilo = round(flo * a + b)
        ihi = round(fhi * a + b)
        # Ensure that if the codomain has only one element, the domain also has
        # only one element that is the average of flo and fhi.
        if ilo == ihi:
            avg = (fhi + flo) / 2
            flo, fhi = avg, avg
        _domain = ContinuousInterval(flo, fhi)
        _codomain = DiscreteInterval(ilo, ihi)
        # Set all the attributes.
        self._a = a
        self._b = b
        self._domain = _domain
        self._codomain = _codomain
        if _inverse is None:
            _inverse = InverseDiscretization(_codomain, _domain, _inverse=self)
            assert _inverse.codomain == self._domain
        self._inverse = _inverse

    @property
    def domain(self) -> ContinuousInterval:
        return self._domain

    @property
    def codomain(self) -> DiscreteInterval:
        return self._codomain

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def inverse(self):
        return self._inverse

    @property
    def states(self):
        ilo, ihi = self.codomain
        return ihi - ilo + 1

    @property
    def eps(self):
        flo, fhi = self.domain
        n = self.states - 1
        return (fhi - flo) / n if n > 0 else 0.0

    def __call__(self, x: float):
        a, b = self.a, self.b
        lo, hi = self.domain
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        return round(a * x + b)

    def __repr__(self) -> str:
        flo, fhi = self.domain
        ilo, ihi = self.codomain
        if self.a < 0:
            ilo, ihi = ihi, ilo
        return f"Discretization(({flo}, {fhi}), ({ilo}, {ihi}))"

    def __eq__(self, other):
        if not isinstance(other, Discretization):
            return False
        return (self.domain == other.domain) and (self.codomain == other.codomain)


class InverseDiscretization():
    "A linear mapping from integers to floating-point values."
    _domain: DiscreteInterval
    _codomain: ContinuousInterval
    _a: float
    _b: float
    _inverse: Discretization

    def __init__(
            self,
            domain: tuple[int, int],
            codomain: tuple[float, float],
            *,
            _inverse: Discretization | None = None):
        # Normalize the domain.
        if domain[1] < domain[0]:
            ihi, ilo = domain
            f2, f1 = codomain
        else:
            (ilo, ihi) = domain
            (f1, f2) = codomain
        # Compute the coefficients.
        dy = f2 - f1
        dx = ihi - ilo
        a = dy / dx if dx > 0.0 else 0.0
        # Normalize the codomain.
        if f2 < f1:
            flo, fhi = f2, f1
            a = -a
        else:
            flo, fhi = f1, f2
        b = flo - a * ilo
        # Ensure that if the domain has only one element, the codomain also has
        # only one element that is the average of flo and fhi.
        if ilo == ihi:
            avg = (fhi + flo) / 2
            flo, fhi = avg, avg
        # Set all the attributes.
        self._a = a
        self._b = b
        self._domain = DiscreteInterval(ilo, ihi)
        self._codomain = ContinuousInterval(flo, fhi)
        if _inverse is None:
            _inverse = Discretization(codomain, domain, _inverse=self)
        self._inverse = _inverse

    @property
    def domain(self) -> DiscreteInterval:
        return self._domain

    @property
    def codomain(self) -> ContinuousInterval:
        return self._codomain

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    @property
    def inverse(self) -> Discretization:
        return self._inverse

    def __call__(self, x: int):
        a, b = self.a, self.b
        ilo, ihi = self.domain
        if x < ilo:
            x = ilo
        elif x > ihi:
            x = ihi
        return a * x + b

    def __repr__(self) -> str:
        ilo, ihi = self.domain
        flo, fhi = self.codomain
        if self.a < 0:
            flo, fhi = fhi, flo
        return f"InverseDiscretization(({ilo}, {ihi}), ({flo}, {fhi}))"

    def __eq__(self, other):
        if not isinstance(other, InverseDiscretization):
            return False
        return (self.domain == other.domain) and (self.codomain == other.codomain)


