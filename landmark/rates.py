"""Dane referencyjne LANDMARK i uśrednianie maxwellowskie przekrojów czynnych.

LANDMARK podaje TABELE współczynników szybkości dla ksenonu:
    k_iz(ε) — współczynnik szybkości jonizacji            [m³/s]
    K(ε)    — współczynnik strat energii zderzeniowych    [eV·m³/s]
gdzie ε to ŚREDNIA energia elektronów, a tabele wyprowadzono zakładając
rozkład MAXWELLOWSKI (EEDF). Dla maxwelliana ε = (3/2)·T_e, więc T_e = 2ε/3.

Żeby porównać mój kinetyczny model zderzeń (oparty na przekrojach czynnych)
z tymi danymi, uśredniam przekroje po rozkładzie maxwellowskim:

    k(T) = ∫₀^∞ σ(E)·v(E)·f(E) dE ,   f(E) = (2/√π)·T^(-3/2)·√E·exp(−E/T)

To jedyna część benchmarku LANDMARK, dla której istnieją twarde dane liczbowe
(a nie tylko wykresy), więc jedyna dająca porównanie w pełni ilościowe.
"""

import os
import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON
from hall_pic import cross_sections as xs

_CSV = os.path.join(os.path.dirname(__file__), "landmark_rates.csv")


def load_landmark_tables():
    """Zwraca (eps_eV, k_iz, K) z tabel LANDMARK."""
    data = np.loadtxt(_CSV, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]


def maxwellian_rate(sigma_fn, T_eV, n_grid=4000, n_T=30.0):
    """⟨σv⟩ dla maxwellowskiego EEDF o temperaturze T_eV (skalar albo wektor).

    Całkowanie po energii od 0 do n_T·T metodą trapezów na gęstej siatce.
    """
    T = np.atleast_1d(np.asarray(T_eV, dtype=float))
    out = np.zeros_like(T)
    for i, t in enumerate(T):
        if t <= 0:
            continue
        E = np.linspace(1e-4, n_T * t, n_grid)
        f = (2.0 / np.sqrt(np.pi)) * t**-1.5 * np.sqrt(E) * np.exp(-E / t)
        v = np.sqrt(2.0 * E * E_CHARGE / M_ELECTRON)
        out[i] = np.trapezoid(sigma_fn(E) * v * f, E)
    return out if out.size > 1 else float(out[0])


def my_k_iz(eps_mean_eV):
    """Mój k_iz uśredniony maxwellowsko, indeksowany ŚREDNIĄ energią ε."""
    T = 2.0 / 3.0 * np.asarray(eps_mean_eV, dtype=float)
    return maxwellian_rate(xs.sigma_ionization_e, T)


def my_K(eps_mean_eV):
    """Mój współczynnik strat energii: Σ próg·⟨σv⟩ po kanałach nieelastycznych.

    Odpowiada definicji LANDMARK (człon strat n·N·K w równaniu energii).
    Uwzględnia jonizację i wzbudzenie; strata elastyczna (~2m/M) jest pomijalna.
    """
    T = 2.0 / 3.0 * np.asarray(eps_mean_eV, dtype=float)
    k_iz = maxwellian_rate(xs.sigma_ionization_e, T)
    k_ex = maxwellian_rate(xs.sigma_excitation_e, T)
    return xs.E_ION_XE * k_iz + xs.E_EXC_XE * k_ex


def landmark_interp(eps_query, eps_tab, val_tab):
    """Interpolacja liniowa tabeli LANDMARK w zadanych punktach ε."""
    return np.interp(eps_query, eps_tab, val_tab)
