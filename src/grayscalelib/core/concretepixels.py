from dataclasses import dataclass
from typing import TypeVar

from grayscalelib.core.discretization import Discretization
from grayscalelib.core.pixels import Pixels, PixelsInitializer


class ConcretePixels(Pixels):
    _shape: tuple[int, ...]
    _discretization: Discretization

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def discretization(self) -> Discretization:
        return self._discretization


CP = TypeVar("CP", bound=ConcretePixels)


@dataclass(frozen=True)
class ConcretePixelsInitializer(PixelsInitializer[CP]):
    shape: tuple[int, ...]
    discretization: Discretization

    def initialize(self, /, instance: CP):
        instance._shape = self.shape
        instance._discretization = self.discretization
