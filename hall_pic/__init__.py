"""Symulacja plazmy silnika Halla w jednym wymiarze osiowym.

Sledzimy pelny wektor predkosci czastek, dokladamy zewnetrzny obwod
zasilajacy, zderzenia z gazem, adaptacyjne dzielenie czastek oraz podglad
przebiegu na zywo. Ten plik udostepnia dwie najwazniejsze rzeczy, ktorych
uzywa reszta programu: konfiguracje oraz sama symulacje.
"""

from .config import Config
from .simulation import Simulation

__all__ = ["Config", "Simulation"]
