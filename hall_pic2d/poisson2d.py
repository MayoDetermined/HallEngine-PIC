"""Wyznaczanie potencjału i pola elektrycznego na płaszczyźnie.

Zadanie jest takie samo jak w wersji jednowymiarowej, tylko na płaszczyźnie:
mając rozkład ładunku, szukamy potencjału zgodnego z warunkami na brzegach.
Wzdłuż osi kanału potencjał jest zadany na obu końcach: przy anodzie równy
napięciu z obwodu, przy katodzie zerowy. W drugim kierunku zależy to od
wybranej płaszczyzny. Kiedy kierunek zawija się w sobie, obowiązuje warunek
okresowy, a kiedy ograniczają go ścianki, potencjał jest przy nich zerowy.

Sztuczka polega na tym, że drugi kierunek można rozłożyć na niezależne fale
i policzyć każdą z nich osobno. Po takim rozłożeniu dla każdej fali zostaje
proste zadanie wzdłuż osi kanału, które rozwiązujemy szybko i od razu dla
wszystkich fal naraz. Całość korzysta wyłącznie z podstawowych operacji na
tablicach.
"""

import numpy as np

from hall_pic.constants import EPS0


class Poisson2D:
    """Liczy potencjał i pole na płaszczyźnie, dla obu układów kanału."""

    def __init__(self, cfg):
        """Przygotowuje rozkład na fale w drugim kierunku i stałe zadania wzdłuż osi."""
        self.cfg = cfg
        self.periodic = (cfg.x2_bc == "periodic")
        self.N1 = cfg.N1
        self.N2 = cfg.N2
        self.h1 = cfg.h1
        self.h2 = cfg.h2
        self.n1 = cfg.N1 - 1            # tyle jest niewiadomych wzdłuż osi, bez dwóch brzegów

        if self.periodic:
            self.nk = cfg.N2                       # przy warunku okresowym niewiadome są wszystkie węzły
            k = np.arange(self.nk)
            self.lam = (2.0 * np.cos(2.0 * np.pi * k / self.N2) - 2.0) / self.h2**2
        else:
            self.nk = cfg.N2 - 1                   # przy ściankach liczą się tylko węzły wewnętrzne
            k = np.arange(1, self.N2)
            self.lam = (2.0 * np.cos(np.pi * k / self.N2) - 2.0) / self.h2**2
            # Macierz zamieniająca wartości na fale i z powrotem, potrzebna, gdy
            # drugi kierunek ograniczają ścianki.
            j = np.arange(1, self.N2)
            self.S = np.sin(np.pi * np.outer(j, k) / self.N2)
            self.dst_scale = 2.0 / self.N2

        # Stałe zadania wzdłuż osi. Skrajne współczynniki są jednakowe, a środkowy
        # zależy od tego, którą falę akurat liczymy.
        self.a = 1.0 / self.h1**2
        self.c = 1.0 / self.h1**2
        self.b = -2.0 / self.h1**2 + self.lam

    # Zamiana między wartościami a falami w drugim kierunku.
    def _fwd(self, f):
        """Rozkłada dane w drugim kierunku na fale."""
        if self.periodic:
            return np.fft.fft(f, axis=1)
        return f @ self.S

    def _inv(self, F):
        """Składa fale z powrotem w zwykłe wartości w drugim kierunku."""
        if self.periodic:
            return np.real(np.fft.ifft(F, axis=1))
        return (F @ self.S) * self.dst_scale

    # Rozwiązanie zadania wzdłuż osi, od razu dla wszystkich fal.
    def _solve_tridiag(self, rhs):
        """Rozwiązuje niezależne zadanie wzdłuż osi kanału dla każdej fali osobno."""
        n1 = self.n1
        a, c, b = self.a, self.c, self.b
        cp = np.empty((n1, self.nk), dtype=rhs.dtype)
        dp = np.empty((n1, self.nk), dtype=rhs.dtype)
        cp[0] = c / b
        dp[0] = rhs[0] / b
        for i in range(1, n1):
            m = b - a * cp[i - 1]
            cp[i] = c / m
            dp[i] = (rhs[i] - a * dp[i - 1]) / m
        x = np.empty((n1, self.nk), dtype=rhs.dtype)
        x[-1] = dp[-1]
        for i in range(n1 - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    # Główne wywołanie.
    def solve(self, rho, V_anode):
        """Zwraca potencjał i obie składowe pola dla podanego rozkładu ładunku.

        Napięcie anody podajemy z zewnątrz, bo narzuca je obwód.
        """
        cfg = self.cfg
        n2n = cfg.n2_nodes

        # Prawa strona dla węzłów wewnętrznych wzdłuż osi.
        src = -rho[1:self.N1] / EPS0
        if not self.periodic:
            src = src[:, 1:self.N2]                      # przy ściankach bierzemy tylko wnętrze

        rhs = self._fwd(src)

        # Wkład od znanego potencjału na anodzie. Anoda ma to samo napięcie na
        # całej szerokości, a katoda jest zerowa, więc nie dokłada nic.
        Va = np.full(n2n, float(V_anode))
        if not self.periodic:
            Va = Va[1:self.N2]
        Va_hat = self._fwd(Va[None, :])[0]
        rhs[0] = rhs[0] - Va_hat / self.h1**2

        # Rozwiązanie dla każdej fali, a potem złożenie fal z powrotem.
        phi_hat = self._solve_tridiag(rhs)
        phi_int = self._inv(phi_hat)

        phi = np.zeros((self.N1 + 1, n2n))
        phi[0, :] = V_anode
        phi[-1, :] = 0.0
        if self.periodic:
            phi[1:self.N1, :] = phi_int
        else:
            phi[1:self.N1, 1:self.N2] = phi_int
            phi[1:self.N1, 0] = 0.0                      # potencjał przy ściankach jest zerowy
            phi[1:self.N1, -1] = 0.0

        E1, E2 = self._gradient(phi)
        return phi, E1, E2

    def _gradient(self, phi):
        """Liczy pole elektryczne z potencjału jako jego nachylenie ze znakiem minus.

        W kierunku, który zawija się w sobie, sąsiada zza brzegu bierzemy z
        drugiego końca.
        """
        E1 = np.empty_like(phi)
        E1[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * self.h1)
        E1[0, :] = -(phi[1, :] - phi[0, :]) / self.h1
        E1[-1, :] = -(phi[-1, :] - phi[-2, :]) / self.h1

        E2 = np.empty_like(phi)
        if self.periodic:
            E2[:, :] = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * self.h2)
        else:
            E2[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * self.h2)
            E2[:, 0] = -(phi[:, 1] - phi[:, 0]) / self.h2
            E2[:, -1] = -(phi[:, -1] - phi[:, -2]) / self.h2
        return E1, E2
