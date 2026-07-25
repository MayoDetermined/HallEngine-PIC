"""Rozwiązanie równania Poissona 1D (elektrostatyka) z warunkami Dirichleta.

    d^2 phi / dx^2 = -rho / eps0
    phi(x=0)  = V_anode   (napięcie na kondensatorze obwodu, dynamiczne)
    phi(x=L)  = 0         (katoda / masa)

Solver trójdiagonalny (Thomas). Macierz jest stała, więc rozkładamy ją raz.
Pole E = -d phi/dx liczone różnicami centralnymi na węzłach.
"""

import numpy as np

from .constants import EPS0


class PoissonSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.Nx = cfg.Nx
        self.dx = cfg.dx
        n_int = self.Nx - 1          # liczba węzłów wewnętrznych (1..Nx-1)
        self.n_int = n_int
        # Macierz operatora dla węzłów wewnętrznych (stała w czasie):
        #   (-phi_{i-1} + 2 phi_i - phi_{i+1}) = rho_i*dx^2/eps0 (+ BC)
        M = (np.diag(2.0 * np.ones(n_int))
             + np.diag(-1.0 * np.ones(n_int - 1), 1)
             + np.diag(-1.0 * np.ones(n_int - 1), -1))
        # Prekomputowana odwrotność -> solve = jeden matvec BLAS (szybkie).
        self.Minv = np.linalg.inv(M)

    def solve(self, rho, V_anode):
        """rho: gęstość ładunku na węzłach [C/m^3] (len Nx+1). Zwraca phi, E (len Nx+1)."""
        dx2 = self.dx * self.dx
        # prawa strona: rho_i * dx^2 / eps0, z korektą brzegów Dirichleta
        d = rho[1:self.Nx] * dx2 / EPS0
        d = d.astype(float).copy()
        d[0] += V_anode      # phi_0 = V_anode przeniesione na prawą stronę
        d[-1] += 0.0         # phi_Nx = 0

        phi_int = self.Minv @ d

        phi = np.empty(self.Nx + 1)
        phi[0] = V_anode
        phi[-1] = 0.0
        phi[1:self.Nx] = phi_int

        # E = -dphi/dx (centralne w środku, jednostronne na brzegach)
        E = np.empty_like(phi)
        E[1:-1] = -(phi[2:] - phi[:-2]) / (2.0 * self.dx)
        E[0] = -(phi[1] - phi[0]) / self.dx
        E[-1] = -(phi[-1] - phi[-2]) / self.dx
        return phi, E
