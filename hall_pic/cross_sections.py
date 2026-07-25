"""Przekroje czynne dla ksenonu (fity analityczne, w m^2, energia w eV).

To są UPROSZCZONE, gładkie przybliżenia inżynierskie — wystarczające do
demonstracji metody null-MCC i formowania wiązek RE, ale do publikacji należy
podstawić dane tablicowe (np. LXCat / Biagi). Funkcje są wektorowe (NumPy).
"""

import numpy as np

from .constants import E_CHARGE, M_ELECTRON, M_XENON

# Progi energetyczne [eV]
E_EXC_XE = 8.32     # próg wzbudzenia (uśredniony)
E_ION_XE = 12.13    # próg jonizacji


def _safe_eps(eps_eV):
    return np.maximum(eps_eV, 1e-3)


# Skalowanie przekroju elastycznego. Pierwotny fit dawał sigma ~3x za duzo
# wzgledem DWOCH niezaleznych referencji: stalego k_m = 2.5e-13 m^3/s z
# LANDMARK oraz danych literaturowych dla Xe. Zawyzony przekroj elastyczny
# nadmiernie tlumi ped elektronow, czyli sztucznie gasi wiazki RE.
# Wspolczynnik dobrany tak, by <sigma*v> przy eps = 10 eV = 2.5e-13 m^3/s.
_EL_SCALE = 0.2688


def sigma_elastic_e(eps_eV):
    """Elastyczne e-Xe. Model z minimum Ramsauera, skalibrowany (patrz _EL_SCALE)."""
    eps = _safe_eps(eps_eV)
    # Ramsauer-podobny kształt: minimum ~0.6 eV, wzrost, potem spadek ~1/eps
    ramsauer = 1.2e-19 * (eps / (eps + 0.5)) ** 2
    high = 6.0e-19 / (1.0 + eps / 30.0)
    return _EL_SCALE * (ramsauer + high)


# --- Parametry SKALIBROWANE względem tabel LANDMARK -------------------
# Postać: sigma(E) = A * ln(u)/u^c * (1 - 1/u)^b,  u = E/E_prog
# Dopasowane tak, by uśredniony maxwellowsko współczynnik szybkości
# odtwarzał tabele LANDMARK k_iz(eps) i K(eps) (patrz landmark/calibrate.py).
# Kontrola niezależna: szczyt sigma_iz = 5.22e-20 m^2 zgadza się z wartością
# eksperymentalną dla Xe (~5.2e-20 m^2 przy ~100 eV).
_IZ_A, _IZ_B, _IZ_C = 9.184486e-20, 0.4091, 0.6000
_EX_A, _EX_B, _EX_C = 1.636412e-19, 0.9364, 1.1000


def sigma_excitation_e(eps_eV):
    """Wzbudzenie e-Xe (jeden efektywny poziom), skalibrowane do LANDMARK K."""
    eps = _safe_eps(eps_eV)
    sig = np.zeros_like(eps)
    mask = eps > E_EXC_XE
    u = eps[mask] / E_EXC_XE
    sig[mask] = _EX_A * np.log(u) / u ** _EX_C * (1.0 - 1.0 / u) ** _EX_B
    return np.maximum(sig, 0.0)


def sigma_ionization_e(eps_eV):
    """Jonizacja e-Xe, skalibrowana do tabeli LANDMARK k_iz."""
    eps = _safe_eps(eps_eV)
    sig = np.zeros_like(eps)
    mask = eps > E_ION_XE
    u = eps[mask] / E_ION_XE
    sig[mask] = _IZ_A * np.log(u) / u ** _IZ_C * (1.0 - 1.0 / u) ** _IZ_B
    return np.maximum(sig, 0.0)


def sigma_total_e(eps_eV):
    return (sigma_elastic_e(eps_eV)
            + sigma_excitation_e(eps_eV)
            + sigma_ionization_e(eps_eV))


# ---------------- Jony ----------------

def sigma_cex_ion(eps_eV):
    """Wymiana ładunku Xe+ - Xe (charge exchange). Fit log-liniowy."""
    eps = _safe_eps(eps_eV)
    # sigma = (a - b ln E)^2 [10^-20 m^2] -> tu w m^2
    a, b = 87.3e-20 ** 0.5, 13.6e-20 ** 0.5  # trzymamy dodatnią bazę
    val = (55.0 - 3.0 * np.log(eps)) * 1e-20
    return np.maximum(val, 1e-20)


def sigma_elastic_ion(eps_eV):
    """Elastyczne (rozpraszanie) Xe+ - Xe, zbliżone do CEX rzędu wielkości."""
    eps = _safe_eps(eps_eV)
    return np.maximum((45.0 - 2.5 * np.log(eps)) * 1e-20, 1e-20)


def sigma_total_ion(eps_eV):
    return sigma_cex_ion(eps_eV) + sigma_elastic_ion(eps_eV)


# ---------------- narzędzia energia<->prędkość ----------------

def speed_from_energy_e(eps_eV):
    return np.sqrt(2.0 * eps_eV * E_CHARGE / M_ELECTRON)


def energy_eV_from_speed_e(v):
    return 0.5 * M_ELECTRON * v ** 2 / E_CHARGE


def energy_eV_from_speed_ion(v):
    return 0.5 * M_XENON * v ** 2 / E_CHARGE
