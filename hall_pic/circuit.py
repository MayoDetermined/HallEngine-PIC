"""Zewnętrzny obwód RLC z kondensatorem, sprzężony z wyładowaniem PIC.

Topologia:

    V_ps --[R]--[L_circ]--( węzeł anody, napięcie V_C )---||--- katoda(0 V)
                                    |
                                   [C]  (do masy)

Zmienne stanu:
    I_L  — prąd w gałęzi zasilacz-R-L                      [A]
    V_C  — napięcie na kondensatorze = potencjał anody     [V]

Równania (Kirchhoff):
    L dI_L/dt = V_ps - R*I_L - V_C
    C dV_C/dt = I_L - I_d

gdzie I_d to prąd wyładowania pobierany przez plazmę na anodzie
(prąd przewodzenia = ładunek zebrany na anodzie / dt, znak dodatni gdy
prąd płynie z obwodu do plazmy).

Całkujemy jawnym RK4 z podkrokami (circuit_substeps), traktując I_d jako
stałe w obrębie jednego kroku PIC.
"""

import numpy as np


class Circuit:
    def __init__(self, cfg):
        self.cfg = cfg
        self.V_ps = cfg.V_ps
        self.R = cfg.R_circ
        self.Lc = cfg.L_circ
        self.C = cfg.C_circ
        self.I_L = cfg.I_L_init
        self.V_C = cfg.V_C_init

    def _deriv(self, I_L, V_C, I_d):
        dI = (self.V_ps - self.R * I_L - V_C) / self.Lc
        dV = (I_L - I_d) / self.C
        return dI, dV

    def advance(self, I_d, dt):
        """Całkuje obwód o dt przy stałym prądzie wyładowania I_d."""
        n = self.cfg.circuit_substeps
        h = dt / n
        I_L, V_C = self.I_L, self.V_C
        for _ in range(n):
            k1 = self._deriv(I_L, V_C, I_d)
            k2 = self._deriv(I_L + 0.5*h*k1[0], V_C + 0.5*h*k1[1], I_d)
            k3 = self._deriv(I_L + 0.5*h*k2[0], V_C + 0.5*h*k2[1], I_d)
            k4 = self._deriv(I_L + h*k3[0], V_C + h*k3[1], I_d)
            I_L += (h/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            V_C += (h/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        self.I_L, self.V_C = I_L, V_C
        # zabezpieczenie numeryczne
        if not np.isfinite(self.V_C):
            self.V_C = 0.0
        return self.V_C
