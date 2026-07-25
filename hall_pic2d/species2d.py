"""Kontener superczątstek 2D3V (Structure-of-Arrays).

Położenie: (x1, x2). Prędkość: (v1, v2, v3) — trzecia składowa poza płaszczyzną.
Waga w [1/m] — cząstki rzeczywiste na metr kierunku ignorowanego (patrz config2d).
"""

import numpy as np

from hall_pic.constants import E_CHARGE


class Species2D:
    def __init__(self, name, charge, mass, capacity=200000):
        self.name = name
        self.charge = charge
        self.mass = mass
        self.N = 0
        self._cap = capacity
        for a in ("x1", "x2", "v1", "v2", "v3", "w"):
            setattr(self, a, np.zeros(capacity))

    _FIELDS = ("x1", "x2", "v1", "v2", "v3", "w")

    def _ensure(self, extra):
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
        if not np.any(kill):
            return
        idx = np.nonzero(~kill)[0]
        m = idx.size
        for a in self._FIELDS:
            arr = getattr(self, a)
            arr[:m] = arr[idx]
        self.N = m

    # widoki aktywnej części
    @property
    def ax1(self): return self.x1[:self.N]
    @property
    def ax2(self): return self.x2[:self.N]
    @property
    def av1(self): return self.v1[:self.N]
    @property
    def av2(self): return self.v2[:self.N]
    @property
    def av3(self): return self.v3[:self.N]
    @property
    def aw(self):  return self.w[:self.N]

    def speed2(self):
        return self.av1**2 + self.av2**2 + self.av3**2

    def kinetic_energy_eV(self):
        return 0.5 * self.mass * self.speed2() / E_CHARGE
