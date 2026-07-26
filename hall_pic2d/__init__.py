"""Wersja symulacji silnika Halla w dwóch wymiarach przestrzennych.

Rdzeń fizyczny jest ten sam co w wersji jednowymiarowej, ale obliczenia dzieją
się teraz na płaszczyźnie i można wybrać jedną z dwóch płaszczyzn kanału. Ten
plik udostępnia dwie najważniejsze rzeczy: konfigurację oraz samą symulację.
"""

from .config2d import Config2D
from .simulation2d import Simulation2D

__all__ = ["Config2D", "Simulation2D"]
