"""Wyznaczanie potencjału i pola elektrycznego wzdłuż kanału.

Mając rozkład ładunku, szukamy potencjału tak, by na anodzie był on równy
napięciu narzuconemu przez obwód, a na katodzie równy zeru. Zagadnienie
sprowadza się do prostego układu równań na wewnętrznych węzłach siatki.
Ponieważ jego macierz nie zmienia się w czasie, odwracamy ją raz na początku,
a potem każdy krok to już tylko jedno mnożenie. Pole elektryczne dostajemy
z potencjału zwykłą różnicą sąsiednich węzłów.
"""

import numpy as np

from .constants import EPS0


class PoissonSolver:
    """Liczy potencjał i pole elektryczne na siatce kanału.

    Macierz układu przygotowujemy i odwracamy jednorazowo w konstruktorze,
    dzięki czemu pojedyncze wywołanie jest bardzo szybkie.
    """

    def __init__(self, cfg):
        """Buduje macierz układu dla węzłów wewnętrznych i od razu ją odwraca."""
        self.cfg = cfg
        self.Nx = cfg.Nx
        self.dx = cfg.dx
        n_int = self.Nx - 1          # tyle jest węzłów w środku kanału, bez dwóch brzegowych
        self.n_int = n_int
        # Macierz łączy każdy węzeł z dwoma sąsiadami i nie zależy od czasu,
        # więc możemy ją zbudować raz na zawsze.
        M = (np.diag(2.0 * np.ones(n_int))
             + np.diag(-1.0 * np.ones(n_int - 1), 1)
             + np.diag(-1.0 * np.ones(n_int - 1), -1))
        # Odwracamy ją tu, żeby później każde rozwiązanie było jednym mnożeniem.
        self.Minv = np.linalg.inv(M)

    def solve(self, rho, V_anode):
        """Zwraca potencjał i pole elektryczne dla podanego rozkładu ładunku.

        Napięcie anody podajemy z zewnątrz, bo narzuca je obwód. Wynik obejmuje
        wszystkie węzły, łącznie z brzegowymi.
        """
        dx2 = self.dx * self.dx
        # Prawa strona układu. Do pierwszego i ostatniego równania trzeba
        # dołożyć wkład od znanych wartości na brzegach.
        d = rho[1:self.Nx] * dx2 / EPS0
        d = d.astype(float).copy()
        d[0] += V_anode      # na anodzie potencjał jest znany, przenosimy go na prawą stronę
        d[-1] += 0.0         # na katodzie potencjał jest zerowy

        phi_int = self.Minv @ d

        phi = np.empty(self.Nx + 1)
        phi[0] = V_anode
        phi[-1] = 0.0
        phi[1:self.Nx] = phi_int

        # Pole to nachylenie potencjału ze znakiem minus. W środku bierzemy
        # różnicę obu sąsiadów, a na samych brzegach różnicę jednostronną.
        E = np.empty_like(phi)
        E[1:-1] = -(phi[2:] - phi[:-2]) / (2.0 * self.dx)
        E[0] = -(phi[1] - phi[0]) / self.dx
        E[-1] = -(phi[-1] - phi[-2]) / self.dx
        return phi, E
