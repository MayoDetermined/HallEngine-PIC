"""Pojemnik na czastki modelowe w wersji dwuwymiarowej.

Czastka ma teraz dwie współrzędne położenia, bo porusza się po płaszczyźnie,
oraz pełne trzy składowe prędkości, z których jedna wychodzi poza tę
płaszczyznę. Wagę liczymy inaczej niż w wersji jednowymiarowej, bo pomijamy
jeden z kierunków przestrzeni; szczegóły opisano przy konfiguracji. Poza tym
pojemnik działa tak samo: dane w osobnych tablicach, rosnących w miarę potrzeb.
"""

import numpy as np

from hall_pic.constants import E_CHARGE


class Species2D:
    """Zbiór czastek modelowych na płaszczyźnie, o wspólnym ładunku i masie."""

    def __init__(self, name, charge, mass, capacity=200000):
        """Tworzy pusty gatunek o zadanym ładunku i masie pojedynczej czastki."""
        self.name = name
        self.charge = charge
        self.mass = mass
        self.N = 0
        self._cap = capacity
        for a in ("x1", "x2", "v1", "v2", "v3", "w"):
            setattr(self, a, np.zeros(capacity))

    _FIELDS = ("x1", "x2", "v1", "v2", "v3", "w")

    def _ensure(self, extra):
        """Upewnia się, że zmieści się jeszcze tyle nowych czastek, i w razie potrzeby powiększa tablice."""
        need = self.N + extra
        if need <= self._cap:
            return
        newcap = max(need, int(self._cap * 2))
        for a in self._FIELDS:
            arr = getattr(self, a)
            new = np.zeros(newcap)
            new[:self.N] = arr[:self.N]
            setattr(self, a, new)
        self._cap = newcap

    def add(self, x1, x2, v1, v2, v3, w):
        """Dopisuje nowe czastki na końcu. Argumenty mogą być tablicami lub liczbami."""
        x1 = np.atleast_1d(x1)
        n = x1.size
        if n == 0:
            return
        self._ensure(n)
        s = slice(self.N, self.N + n)
        self.x1[s] = x1
        self.x2[s] = np.atleast_1d(x2)
        self.v1[s] = np.atleast_1d(v1)
        self.v2[s] = np.atleast_1d(v2)
        self.v3[s] = np.atleast_1d(v3)
        self.w[s] = np.atleast_1d(w)
        self.N += n

    def remove_mask(self, kill):
        """Usuwa czastki zaznaczone w masce i upycha pozostałe od początku tablic."""
        if not np.any(kill):
            return
        idx = np.nonzero(~kill)[0]
        m = idx.size
        for a in self._FIELDS:
            arr = getattr(self, a)
            arr[:m] = arr[idx]
        self.N = m

    # Poniższe własności zwracają widok samych aktywnych czastek, czyli
    # początkowy wycinek każdej tablicy, bez kopiowania danych.
    @property
    def ax1(self):
        """Położenia aktywnych czastek wzdłuż osi kanału."""
        return self.x1[:self.N]

    @property
    def ax2(self):
        """Położenia aktywnych czastek wzdłuż drugiego kierunku płaszczyzny."""
        return self.x2[:self.N]

    @property
    def av1(self):
        """Prędkości aktywnych czastek wzdłuż osi kanału."""
        return self.v1[:self.N]

    @property
    def av2(self):
        """Prędkości aktywnych czastek wzdłuż drugiego kierunku płaszczyzny."""
        return self.v2[:self.N]

    @property
    def av3(self):
        """Prędkości aktywnych czastek w kierunku wychodzącym poza płaszczyznę."""
        return self.v3[:self.N]

    @property
    def aw(self):
        """Wagi aktywnych czastek."""
        return self.w[:self.N]

    def speed2(self):
        """Kwadrat szybkości każdej aktywnej czastki, czyli suma kwadratów prędkości."""
        return self.av1**2 + self.av2**2 + self.av3**2

    def kinetic_energy_eV(self):
        """Energia ruchu każdej aktywnej czastki, wyrażona w elektronowoltach."""
        return 0.5 * self.mass * self.speed2() / E_CHARGE
