"""Pojemnik na czastki modelowe jednego gatunku.

Nie śledzimy każdego pojedynczego elektronu czy jonu, bo byłoby ich zbyt
wiele. Zamiast tego każda czastka modelowa zastępuje całą paczkę czastek
rzeczywistych, a liczbę, którą reprezentuje, nazywamy jej wagą. Dzięki
jednakowej definicji wagi ta sama liczba służy zarówno do wyznaczania
gęstości ładunku, jak i do wyznaczania prądu w obwodzie, więc jedno z drugim
się zgadza. Dane trzymamy w osobnych tablicach dla położenia, prędkości i
wagi, co przyspiesza obliczenia na całych paczkach naraz.
"""

import numpy as np


class Species:
    """Zbiór czastek modelowych o wspólnym ładunku i masie.

    Położenia i prędkości leżą w osobnych tablicach o wspólnej pojemności,
    a pole N mówi, ile z nich jest w danej chwili używanych. Kiedy zabraknie
    miejsca, tablice rosną dwukrotnie.
    """

    def __init__(self, name, charge, mass, capacity=100000):
        """Tworzy pusty gatunek o zadanym ładunku i masie pojedynczej czastki.

        Nazwa służy tylko do rozróżniania w podglądzie, a początkowa pojemność
        rośnie sama w miarę potrzeb.
        """
        self.name = name
        self.charge = charge          # ładunek pojedynczej czastki rzeczywistej
        self.mass = mass              # masa pojedynczej czastki
        self.N = 0                    # ile czastek jest teraz aktywnych
        self._cap = capacity
        self.x = np.zeros(capacity)
        self.vx = np.zeros(capacity)
        self.vy = np.zeros(capacity)
        self.vz = np.zeros(capacity)
        self.w = np.zeros(capacity)   # waga, czyli ile czastek rzeczywistych zastępuje dana czastka modelowa

    # Rozrastanie tablic w miarę potrzeby.
    def _ensure(self, extra):
        """Upewnia się, że zmieści się jeszcze tyle nowych czastek.

        Jeśli brakuje miejsca, przepisuje dane do dwukrotnie większych tablic.
        """
        need = self.N + extra
        if need <= self._cap:
            return
        newcap = max(need, int(self._cap * 2))
        for attr in ("x", "vx", "vy", "vz", "w"):
            arr = getattr(self, attr)
            new = np.zeros(newcap)
            new[:self.N] = arr[:self.N]
            setattr(self, attr, new)
        self._cap = newcap

    def add(self, x, vx, vy, vz, w):
        """Dopisuje nowe czastki na końcu. Argumenty mogą być tablicami lub liczbami."""
        x = np.atleast_1d(x)
        n = x.size
        if n == 0:
            return
        self._ensure(n)
        s = slice(self.N, self.N + n)
        self.x[s] = x
        self.vx[s] = np.atleast_1d(vx)
        self.vy[s] = np.atleast_1d(vy)
        self.vz[s] = np.atleast_1d(vz)
        self.w[s] = np.atleast_1d(w)
        self.N += n

    def remove_mask(self, kill):
        """Usuwa czastki zaznaczone w masce i upycha pozostałe od początku tablic."""
        if not np.any(kill):
            return
        keep = ~kill
        idx = np.nonzero(keep)[0]
        m = idx.size
        for attr in ("x", "vx", "vy", "vz", "w"):
            arr = getattr(self, attr)
            arr[:m] = arr[idx]
        self.N = m

    # Poniższe własności zwracają widok samych aktywnych czastek, czyli
    # początkowy wycinek każdej tablicy, bez kopiowania danych.
    @property
    def ax(self):
        """Położenia aktywnych czastek."""
        return self.x[:self.N]

    @property
    def avx(self):
        """Prędkości aktywnych czastek wzdłuż osi kanału."""
        return self.vx[:self.N]

    @property
    def avy(self):
        """Prędkości aktywnych czastek w drugim kierunku."""
        return self.vy[:self.N]

    @property
    def avz(self):
        """Prędkości aktywnych czastek w trzecim kierunku."""
        return self.vz[:self.N]

    @property
    def aw(self):
        """Wagi aktywnych czastek."""
        return self.w[:self.N]

    def speed2(self):
        """Kwadrat szybkości każdej aktywnej czastki, czyli suma kwadratów prędkości."""
        return self.avx**2 + self.avy**2 + self.avz**2

    def kinetic_energy_eV(self):
        """Energia ruchu każdej aktywnej czastki, wyrażona w elektronowoltach."""
        from .constants import E_CHARGE
        return 0.5 * self.mass * self.speed2() / E_CHARGE
