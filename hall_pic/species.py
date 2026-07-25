"""Kontener superczątstek makro (Structure-of-Arrays) z ważeniem.

Każda superczątstka reprezentuje `w` cząstek rzeczywistych NA JEDNOSTKĘ
PRZEKROJU [1/m^2]. Dzięki temu:
    * gęstość:   n(x) = sum_p w_p * S(x - x_p) / dx           [1/m^3]
    * prąd:      I    = q * A_channel * sum_p w_p / dt        [A]
gdzie S to jądro CIC pierwszego rzędu, a A_channel to pole przekroju kanału.

Ta jednolita definicja wagi zapewnia spójną normalizację ładunku (Poisson)
i prądu (obwód). APR zmienia w_p lokalnie zachowując sum(w_p) w komórce.
"""

import numpy as np


class Species:
    def __init__(self, name, charge, mass, capacity=100000):
        self.name = name
        self.charge = charge          # ładunek pojedynczej cząstki rzeczywistej [C]
        self.mass = mass              # masa [kg]
        self.N = 0                    # liczba aktywnych superczątstek
        self._cap = capacity
        self.x = np.zeros(capacity)
        self.vx = np.zeros(capacity)
        self.vy = np.zeros(capacity)
        self.vz = np.zeros(capacity)
        self.w = np.zeros(capacity)   # waga [1/m^2] (cząstki rzeczywiste na m^2)

    # ---- zarządzanie pojemnością ----
    def _ensure(self, extra):
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
        """Usuwa cząstki gdzie kill==True, kompaktując tablice."""
        if not np.any(kill):
            return
        keep = ~kill
        idx = np.nonzero(keep)[0]
        m = idx.size
        for attr in ("x", "vx", "vy", "vz", "w"):
            arr = getattr(self, attr)
            arr[:m] = arr[idx]
        self.N = m

    # widoki aktywnej części (bez kopiowania)
    @property
    def ax(self):  return self.x[:self.N]
    @property
    def avx(self): return self.vx[:self.N]
    @property
    def avy(self): return self.vy[:self.N]
    @property
    def avz(self): return self.vz[:self.N]
    @property
    def aw(self):  return self.w[:self.N]

    def speed2(self):
        return self.avx**2 + self.avy**2 + self.avz**2

    def kinetic_energy_eV(self):
        from .constants import E_CHARGE
        return 0.5 * self.mass * self.speed2() / E_CHARGE
