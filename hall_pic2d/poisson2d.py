"""Separowalny solver Poissona 2D dla obu geometrii silnika Halla.

    d²φ/dx1² + d²φ/dx2² = -ρ/ε0

Warunki brzegowe:
  * x1 = z (osiowa): Dirichlet na obu końcach — φ(0,·)=V_anoda, φ(L1,·)=0.
  * x2: zależnie od geometrii
      - 'z-theta' : PERIODYCZNY (azymut)          -> transformata FFT
      - 'z-r'     : Dirichlet (ścianki kanału)    -> transformata DST-I

Metoda: transformujemy wzdłuż x2, co diagonalizuje drugą różnicę w tym
kierunku (wartość własna λ_k). Dla każdej mody k zostaje układ trójdiagonalny
wzdłuż z, rozwiązywany ZWEKTORYZOWANYM algorytmem Thomasa (pętla po ~N1,
operacje wektorowe po wszystkich modach naraz). Czysty NumPy, bez scipy.

Wartości własne drugiej różnicy wzdłuż x2:
  periodyczny : λ_k = (2cos(2πk/N2) − 2)/h2²,  k = 0..N2−1
  DST-I       : λ_k = (2cos(πk/N2)  − 2)/h2²,  k = 1..N2−1
"""

import numpy as np

from hall_pic.constants import EPS0


class Poisson2D:
    def __init__(self, cfg):
        self.cfg = cfg
        self.periodic = (cfg.x2_bc == "periodic")
        self.N1 = cfg.N1
        self.N2 = cfg.N2
        self.h1 = cfg.h1
        self.h2 = cfg.h2
        self.n1 = cfg.N1 - 1            # niewiadome wzdłuż z (węzły 1..N1-1)

        if self.periodic:
            self.nk = cfg.N2                       # wszystkie węzły są niewiadomymi
            k = np.arange(self.nk)
            self.lam = (2.0 * np.cos(2.0 * np.pi * k / self.N2) - 2.0) / self.h2**2
        else:
            self.nk = cfg.N2 - 1                   # węzły wewnętrzne 1..N2-1
            k = np.arange(1, self.N2)
            self.lam = (2.0 * np.cos(np.pi * k / self.N2) - 2.0) / self.h2**2
            # macierz DST-I: S[j-1,k-1] = sin(π j k / N2)
            j = np.arange(1, self.N2)
            self.S = np.sin(np.pi * np.outer(j, k) / self.N2)
            self.dst_scale = 2.0 / self.N2         # S @ (S @ f) = (N2/2) f

        # współczynniki trójdiagonalne wzdłuż z (a, c stałe; b zależy od mody)
        self.a = 1.0 / self.h1**2
        self.c = 1.0 / self.h1**2
        self.b = -2.0 / self.h1**2 + self.lam      # (nk,)

    # ---------------- transformaty wzdłuż x2 ----------------
    def _fwd(self, f):
        """f: (n1, n2_nodes) -> (n1, nk) w przestrzeni mod."""
        if self.periodic:
            return np.fft.fft(f, axis=1)
        return f @ self.S                          # DST-I (bez skalowania)

    def _inv(self, F):
        if self.periodic:
            return np.real(np.fft.ifft(F, axis=1))
        return (F @ self.S) * self.dst_scale

    # ---------------- zwektoryzowany Thomas po modach ----------------
    def _solve_tridiag(self, rhs):
        """rhs: (n1, nk). Rozwiązuje niezależny układ trójdiagonalny dla każdej mody."""
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

    # ---------------- główne wywołanie ----------------
    def solve(self, rho, V_anode):
        """rho: (N1+1, n2_nodes) [C/m³]. Zwraca (phi, E1, E2) o tym samym kształcie."""
        cfg = self.cfg
        n2n = cfg.n2_nodes

        # --- prawa strona dla węzłów wewnętrznych wzdłuż z ---
        src = -rho[1:self.N1] / EPS0                     # (n1, n2_nodes)
        if not self.periodic:
            src = src[:, 1:self.N2]                      # tylko węzły wewnętrzne x2

        rhs = self._fwd(src)

        # --- wkład warunków Dirichleta wzdłuż z (anoda / katoda) ---
        # φ(0,·) = V_anode (stałe po x2), φ(L1,·) = 0
        Va = np.full(n2n, float(V_anode))
        if not self.periodic:
            Va = Va[1:self.N2]
        Va_hat = self._fwd(Va[None, :])[0]               # (nk,)
        rhs[0] = rhs[0] - Va_hat / self.h1**2            # katoda = 0 -> brak członu

        # --- rozwiązanie modowe + transformata odwrotna ---
        phi_hat = self._solve_tridiag(rhs)
        phi_int = self._inv(phi_hat)                     # (n1, nk_nodes)

        phi = np.zeros((self.N1 + 1, n2n))
        phi[0, :] = V_anode
        phi[-1, :] = 0.0
        if self.periodic:
            phi[1:self.N1, :] = phi_int
        else:
            phi[1:self.N1, 1:self.N2] = phi_int
            phi[1:self.N1, 0] = 0.0                      # ścianki uziemione
            phi[1:self.N1, -1] = 0.0

        E1, E2 = self._gradient(phi)
        return phi, E1, E2

    def _gradient(self, phi):
        """E = -∇φ. Różnice centralne; x2 periodyczne -> np.roll."""
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
