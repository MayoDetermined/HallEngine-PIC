"""Kalibracja przekrojów czynnych ksenonu względem tabel LANDMARK.

Benchmark wykazał, że oryginalne fity dawały k_iz ~4x za niskie. Zamiast
zostawić rozbieżność, dopasowujemy parametry przekrojów tak, by uśredniony
maxwellowsko współczynnik szybkości odtwarzał tabelę LANDMARK.

Postać dopasowywana (u = E/E_próg):
    σ(E) = A · ln(u)/u^c · (1 − 1/u)^b        dla E > E_próg

Dopasowanie w przestrzeni LOGARYTMICZNEJ (minimalizacja Σ[ln(k_mój/k_ref)]²),
bo współczynniki zmieniają się o rzędy wielkości.

    python -m landmark.calibrate
"""

import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON
from hall_pic import cross_sections as xs
from . import rates


def sigma_form(E, A, b, c, E_th):
    E = np.asarray(E, dtype=float)
    out = np.zeros_like(E)
    m = E > E_th
    u = E[m] / E_th
    out[m] = A * np.log(u) / u**c * (1.0 - 1.0 / u) ** b
    return np.maximum(out, 0.0)


def maxwellian_rate_of(sigma_fn, T_eV, n_grid=3000, n_T=30.0):
    T = np.atleast_1d(np.asarray(T_eV, dtype=float))
    out = np.zeros_like(T)
    for i, t in enumerate(T):
        E = np.linspace(1e-4, n_T * t, n_grid)
        f = (2.0 / np.sqrt(np.pi)) * t**-1.5 * np.sqrt(E) * np.exp(-E / t)
        v = np.sqrt(2.0 * E * E_CHARGE / M_ELECTRON)
        out[i] = np.trapezoid(sigma_fn(E) * v * f, E)
    return out


def fit(E_th, ref_eps, ref_k, label):
    """Siatkowe przeszukanie + zagęszczenie wokół minimum."""
    T = 2.0 / 3.0 * ref_eps

    def cost(A, b, c):
        k = maxwellian_rate_of(lambda E: sigma_form(E, A, b, c, E_th), T)
        good = (k > 0) & (ref_k > 0)
        if good.sum() < 10:
            return np.inf
        return float(np.mean(np.log(k[good] / ref_k[good]) ** 2))

    best = (np.inf, None)
    # zgrubna siatka po kształcie; amplitudę skalujemy analitycznie
    for b in np.linspace(0.2, 2.5, 12):
        for c in np.linspace(0.4, 1.6, 13):
            k1 = maxwellian_rate_of(lambda E: sigma_form(E, 1.0, b, c, E_th), T)
            good = (k1 > 0) & (ref_k > 0)
            if good.sum() < 10:
                continue
            # optymalna amplituda w log-przestrzeni = exp(średnia z ln(ref/k1))
            A = float(np.exp(np.mean(np.log(ref_k[good] / k1[good]))))
            v = cost(A, b, c)
            if v < best[0]:
                best = (v, (A, b, c))

    # zagęszczenie wokół najlepszego
    A0, b0, c0 = best[1]
    for b in np.linspace(max(0.05, b0 - 0.25), b0 + 0.25, 11):
        for c in np.linspace(max(0.1, c0 - 0.15), c0 + 0.15, 11):
            k1 = maxwellian_rate_of(lambda E: sigma_form(E, 1.0, b, c, E_th), T)
            good = (k1 > 0) & (ref_k > 0)
            if good.sum() < 10:
                continue
            A = float(np.exp(np.mean(np.log(ref_k[good] / k1[good]))))
            v = cost(A, b, c)
            if v < best[0]:
                best = (v, (A, b, c))

    v, (A, b, c) = best
    k = maxwellian_rate_of(lambda E: sigma_form(E, A, b, c, E_th), T)
    ratio = k / ref_k
    print(f"\n  {label}:")
    print(f"    A = {A:.6e}   b = {b:.4f}   c = {c:.4f}")
    print(f"    RMS log-błędu = {np.sqrt(v):.4f}"
          f"   (stosunek: mediana {np.median(ratio):.3f}, "
          f"min {ratio.min():.3f}, max {ratio.max():.3f})")
    print(f"    σ w szczycie ≈ {sigma_form(np.linspace(E_th, 500, 2000), A, b, c, E_th).max():.3e} m²")
    return A, b, c


def main():
    print("=" * 72)
    print(" Kalibracja przekrojów czynnych Xe względem tabel LANDMARK")
    print("=" * 72)
    eps_t, kiz_t, K_t = rates.load_landmark_tables()
    m = (eps_t >= 5.0) & (eps_t <= 150.0)
    eps, kiz_ref, K_ref = eps_t[m], kiz_t[m], K_t[m]

    # --- jonizacja: dopasowanie bezpośrednio do k_iz ---
    Ai, bi, ci = fit(xs.E_ION_XE, eps, kiz_ref, "JONIZACJA (do k_iz)")

    # --- wzbudzenie: z reszty K = E_iz·k_iz + E_exc·k_exc ---
    T = 2.0 / 3.0 * eps
    k_iz_fit = maxwellian_rate_of(lambda E: sigma_form(E, Ai, bi, ci, xs.E_ION_XE), T)
    k_exc_target = (K_ref - xs.E_ION_XE * k_iz_fit) / xs.E_EXC_XE
    ok = k_exc_target > 0
    print(f"\n  (punktów z dodatnią resztą na wzbudzenie: {ok.sum()}/{ok.size})")
    Ae, be, ce = fit(xs.E_EXC_XE, eps[ok], k_exc_target[ok], "WZBUDZENIE (z reszty K)")

    print("\n" + "=" * 72)
    print(" Wstaw do hall_pic/cross_sections.py:")
    print("=" * 72)
    print(f"  jonizacja : A={Ai:.6e}, b={bi:.4f}, c={ci:.4f}, E_th={xs.E_ION_XE}")
    print(f"  wzbudzenie: A={Ae:.6e}, b={be:.4f}, c={ce:.4f}, E_th={xs.E_EXC_XE}")


if __name__ == "__main__":
    main()
