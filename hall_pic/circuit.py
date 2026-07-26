"""Zewnętrzny obwód zasilający, powiązany z wyładowaniem w kanale.

Zasilacz podaje napięcie przez opór i cewkę do węzła anody, a równolegle do
samego wyładowania wisi kondensator prowadzący do masy. Śledzimy dwie
wielkości: prąd płynący przez gałąź z cewką oraz napięcie na kondensatorze.
To drugie jest zarazem napięciem anody, więc łączy obwód z resztą symulacji.

Plazma pobiera z anody pewien prąd, który dla obwodu jest zadany z zewnątrz.
W obrębie jednego kroku symulacji traktujemy go jako stały i całkujemy obwód
klasyczną metodą Rungego-Kutty czwartego rzędu, dzieląc krok na kilka
mniejszych dla większej dokładności.
"""

import numpy as np


class Circuit:
    """Model zewnętrznego obwodu zasilającego.

    Pamięta prąd w gałęzi z cewką oraz napięcie na kondensatorze. To napięcie
    jest jednocześnie napięciem anody, którym karmimy solver potencjału.
    """

    def __init__(self, cfg):
        """Przepisuje wartości elementów i stan początkowy z konfiguracji."""
        self.cfg = cfg
        self.V_ps = cfg.V_ps
        self.R = cfg.R_circ
        self.Lc = cfg.L_circ
        self.C = cfg.C_circ
        self.I_L = cfg.I_L_init
        self.V_C = cfg.V_C_init

    def _deriv(self, I_L, V_C, I_d):
        """Zwraca tempo zmian prądu i napięcia dla zadanego stanu obwodu."""
        dI = (self.V_ps - self.R * I_L - V_C) / self.Lc
        dV = (I_L - I_d) / self.C
        return dI, dV

    def advance(self, I_d, dt):
        """Przesuwa stan obwodu o jeden krok, przy zadanym prądzie wyładowania.

        Krok dzielimy na kilka mniejszych, a każdy z nich liczymy metodą
        Rungego-Kutty. Na koniec zwracamy nowe napięcie anody.
        """
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
        # Gdyby coś rozjechało się numerycznie, zabezpieczamy się przed
        # przekazaniem dalej wartości nieskończonej.
        if not np.isfinite(self.V_C):
            self.V_C = 0.0
        return self.V_C
