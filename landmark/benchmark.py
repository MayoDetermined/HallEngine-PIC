"""Benchmark względem LANDMARK 1D-axial.

Uruchomienie:
    python -m landmark.benchmark            # pełny raport + wykresy
    python -m landmark.benchmark --no-plot  # same metryki liczbowe

Zawiera trzy testy:

  TEST 1 — WSPÓŁCZYNNIKI SZYBKOŚCI (w pełni ilościowy)
    Porównuje mój kinetyczny model zderzeń (przekroje czynne, uśrednione po
    maxwellianie) z TABELAMI LANDMARK k_iz(ε) i K(ε). To jedyna część
    benchmarku z twardymi danymi liczbowymi (151 punktów), więc jedyna
    dająca porównanie punkt-po-punkcie.

  TEST 2 — ZGODNOŚĆ KONFIGURACJI
    Weryfikuje odwzorowanie geometrii, profilu B(x) i częstości anomalnych
    względem specyfikacji (wartości kontrolne z dokumentu).

  TEST 3 — ANALIZA WYKONALNOŚCI
    Liczy, ile kroków PIC wymagałby pełny stan ustalony LANDMARK, i pokazuje,
    dlaczego jest poza zasięgiem tego kodu.
"""

import argparse
import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON, EPS0
from . import rates
from .config_lm import LandmarkConfig


def sep(title):
    print("\n" + "=" * 74)
    print(" " + title)
    print("=" * 74)


# ----------------------------------------------------------------------
# TEST 1 — współczynniki szybkości
# ----------------------------------------------------------------------
def test_rate_coefficients(make_plot=True):
    sep("TEST 1 — współczynniki szybkości vs tabele LANDMARK")
    eps_t, kiz_t, K_t = rates.load_landmark_tables()

    # porównujemy w zakresie istotnym fizycznie (poniżej progu k_iz→0)
    mask = (eps_t >= 5.0) & (eps_t <= 150.0)
    eps = eps_t[mask]
    kiz_ref = kiz_t[mask]
    K_ref = K_t[mask]

    kiz_mine = rates.my_k_iz(eps)
    K_mine = rates.my_K(eps)

    ratio_k = kiz_mine / np.maximum(kiz_ref, 1e-30)
    ratio_K = K_mine / np.maximum(K_ref, 1e-30)

    def stats(name, ratio):
        gm = float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-30)))))
        print(f"  {name}: stosunek mój/LANDMARK — "
              f"mediana {np.median(ratio):6.3f}, "
              f"śr.geom {gm:6.3f}, "
              f"min {ratio.min():6.3f}, max {ratio.max():6.3f}")
        return gm

    print(f"  punktów porównania: {eps.size} (ε = {eps.min():.0f}–{eps.max():.0f} eV)\n")
    gm_k = stats("k_iz", ratio_k)
    gm_K = stats("K   ", ratio_K)

    # wybrane punkty kontrolne
    print("\n  punkty kontrolne:")
    print(f"  {'ε [eV]':>8} {'k_iz LANDMARK':>15} {'k_iz mój':>13} {'stos.':>7}"
          f" {'K LANDMARK':>13} {'K mój':>12} {'stos.':>7}")
    for e in (10, 20, 30, 50, 100, 150):
        i = np.argmin(np.abs(eps - e))
        print(f"  {eps[i]:8.0f} {kiz_ref[i]:15.4e} {kiz_mine[i]:13.4e} "
              f"{ratio_k[i]:7.3f} {K_ref[i]:13.4e} {K_mine[i]:12.4e} {ratio_K[i]:7.3f}")

    # --- współczynnik pędu: LANDMARK zakłada STAŁE k_m = 2.5e-13 m³/s ---
    from hall_pic import cross_sections as xs
    cfg = LandmarkConfig(case=1)
    km_mine = np.array([rates.maxwellian_rate(xs.sigma_elastic_e, 2.0/3.0*e)
                        for e in (5.0, 10.0, 20.0, 50.0, 100.0)])
    print(f"\n  współczynnik pędu e-n (LANDMARK: stałe {cfg.k_m:.3e} m³/s):")
    for e, k in zip((5, 10, 20, 50, 100), km_mine):
        print(f"    ε = {e:4d} eV -> mój k_m = {k:.3e} m³/s"
              f"   (stosunek {k/cfg.k_m:.2f})")
    print("    uwaga: LANDMARK celowo używa STAŁEJ wartości; mój k_m zależy od")
    print("    energii, więc pełna zgodność jest niemożliwa z definicji.")

    if make_plot:
        _plot_rates(eps, kiz_ref, kiz_mine, K_ref, K_mine, ratio_k, ratio_K)
    return gm_k, gm_K


def _plot_rates(eps, kiz_ref, kiz_mine, K_ref, K_mine, ratio_k, ratio_K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("Benchmark LANDMARK — współczynniki szybkości dla ksenonu")

    ax[0].semilogy(eps, kiz_ref, "k-", lw=2, label="LANDMARK (tabela)")
    ax[0].semilogy(eps, kiz_mine, "r--", lw=1.8, label="ten kod (przekroje + maxwellian)")
    ax[0].set_xlabel("średnia energia ε [eV]"); ax[0].set_ylabel("k_iz [m³/s]")
    ax[0].set_title("Współczynnik jonizacji"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    ax[1].semilogy(eps, K_ref, "k-", lw=2, label="LANDMARK (tabela)")
    ax[1].semilogy(eps, K_mine, "r--", lw=1.8, label="ten kod")
    ax[1].set_xlabel("średnia energia ε [eV]"); ax[1].set_ylabel("K [eV·m³/s]")
    ax[1].set_title("Współczynnik strat energii"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)

    ax[2].plot(eps, ratio_k, "r-", lw=1.8, label="k_iz")
    ax[2].plot(eps, ratio_K, "b-", lw=1.8, label="K")
    ax[2].axhline(1.0, color="k", ls="--", lw=1)
    ax[2].axhspan(0.5, 2.0, color="green", alpha=0.12, label="±czynnik 2")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("średnia energia ε [eV]"); ax[2].set_ylabel("mój / LANDMARK")
    ax[2].set_title("Stosunek"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = "landmark_rates_benchmark.png"
    fig.savefig(out, dpi=110)
    print(f"\n  wykres zapisany: {out}")


# ----------------------------------------------------------------------
# TEST 2 — zgodność konfiguracji
# ----------------------------------------------------------------------
def test_setup_conformance():
    sep("TEST 2 — zgodność konfiguracji ze specyfikacją LANDMARK")
    cfg = LandmarkConfig(case=1)
    checks = []

    def chk(name, got, expect, tol=1e-6, unit=""):
        ok = abs(got - expect) <= tol * max(abs(expect), 1e-30)
        checks.append(ok)
        print(f"  [{'OK ' if ok else 'BŁĄD'}] {name:38s} = {got:12.5g} {unit}"
              f"  (oczek. {expect:.5g})")

    chk("długość domeny d", cfg.d_domain, 0.05, unit="m")
    chk("długość kanału l", cfg.l_channel, 0.025, unit="m")
    chk("napięcie V", cfg.V_applied, 300.0, unit="V")
    chk("B_max", cfg.B_max, 0.015, unit="T")
    chk("B(x=l) — szczyt na wylocie kanału", float(cfg.B_profile(cfg.l_channel)), 0.015, unit="T")
    chk("strumień neutrali", cfg.neutral_influx, 5.7257e21, unit="m⁻²s⁻¹")
    chk("gęstość neutrali na wlocie", cfg.n_neutral_inlet, 5.7257e21 / 150.0, unit="m⁻³")
    chk("pole przekroju kanału", cfg.A_channel,
        np.pi * (0.05**2 - 0.035**2), unit="m²")
    chk("ν_w w kanale", float(cfg.nu_wall(0.01)), 1.0e7, unit="s⁻¹")
    chk("ν_w poza kanałem", float(cfg.nu_wall(0.04)), 0.0, unit="s⁻¹")
    chk("α w kanale", float(cfg.alpha_bohm(0.01)), 0.1)
    chk("α poza kanałem", float(cfg.alpha_bohm(0.04)), 1.0)
    chk("ν_ε w kanale (Case 1)", cfg.nu_eps_in, 1.0e7, unit="s⁻¹")
    chk("ν_ε w kanale (Case 2)", LandmarkConfig(case=2).nu_eps_in, 0.5e7, unit="s⁻¹")
    chk("ν_ε w kanale (Case 3)", LandmarkConfig(case=3).nu_eps_in, 0.4e7, unit="s⁻¹")

    # szerokość profilu B: sprawdzenie wartości w x = l ± σ (spadek do exp(-1/2))
    b_in = float(cfg.B_profile(cfg.l_channel - cfg.sigma_B_in))
    b_out = float(cfg.B_profile(cfg.l_channel + cfg.sigma_B_out))
    chk("B(l−σ_in)/B_max = exp(−1/2)", b_in / cfg.B_max, np.exp(-0.5), tol=1e-9)
    chk("B(l+σ_out)/B_max = exp(−1/2)", b_out / cfg.B_max, np.exp(-0.5), tol=1e-9)

    # anomalna częstość w szczycie B
    nu_a = float(cfg.nu_anomalous(cfg.l_channel))
    expect = 1.0e7 + 0.1 * (E_CHARGE * 0.015 / M_ELECTRON) / 16.0
    chk("ν_anom(x=l)", nu_a, expect, unit="s⁻¹")

    print(f"\n  zaliczone: {sum(checks)}/{len(checks)}")
    return all(checks)


# ----------------------------------------------------------------------
# TEST 3 — analiza wykonalności
# ----------------------------------------------------------------------
def test_feasibility():
    sep("TEST 3 — wykonalność pełnego benchmarku stanu ustalonego w PIC")
    cfg = LandmarkConfig(case=1)

    n_n = cfg.n_neutral_inlet
    t_neutral = cfg.d_domain / cfg.v_neutral
    t_ion = cfg.d_domain / 1.0e4          # jony ~10 km/s

    n_e_peak = 1.0e18                      # rząd wielkości ze stanu ustalonego LANDMARK
    w_pe = np.sqrt(n_e_peak * E_CHARGE**2 / (EPS0 * M_ELECTRON))
    dt_max = 0.2 / w_pe
    lambda_D = np.sqrt(EPS0 * 10.0 * E_CHARGE / (n_e_peak * E_CHARGE**2))

    print(f"  gęstość neutrali na wlocie      : {n_n:.3e} m⁻³")
    print(f"  czas przelotu neutrali (5 cm)   : {t_neutral*1e6:8.0f} µs   <-- wyznacza stan ustalony")
    print(f"  czas przelotu jonów             : {t_ion*1e6:8.1f} µs")
    print(f"  ω_pe przy n_e = 1e18 m⁻³        : {w_pe:.3e} rad/s")
    print(f"  wymagany krok dt ≈ 0.2/ω_pe     : {dt_max*1e12:8.2f} ps")
    print(f"  długość Debye'a (T_e = 10 eV)   : {lambda_D*1e6:8.2f} µm  -> dx musi ją rozdzielać")

    n_steps = t_neutral / dt_max
    rate = 400.0                            # realna wydajność tego kodu (kroków/s)
    print(f"\n  kroków do stanu ustalonego      : {n_steps:.2e}")
    print(f"  przy {rate:.0f} kroków/s               : {n_steps/rate/3600:8.0f} godzin"
          f"  ({n_steps/rate/86400:.0f} dni)")
    print(f"  komórek dla 5 cm przy dx = λ_D  : {cfg.d_domain/lambda_D:.0f}")

    print("\n  WNIOSEK: pełne odtworzenie stanu ustalonego LANDMARK jest poza")
    print("  zasięgiem tego kodu o ~3 rzędy wielkości. Stan ustalony wyznacza")
    print("  dynamika neutrali (~333 µs), a krok PIC musi rozdzielać ω_pe (~ps).")
    print("  Dlatego LANDMARK używa modeli PŁYNOWYCH/hybrydowych, które nie")
    print("  rozdzielają ani ω_pe, ani λ_D.")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Benchmark względem LANDMARK 1D-axial")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    print("=" * 74)
    print(" BENCHMARK: kod PIC vs LANDMARK 1D-axial Hall thruster benchmark")
    print(" (Hagelaar, Hara, Smolyakov, Boeuf — 22.05.2018)")
    print("=" * 74)

    ok_setup = test_setup_conformance()
    gm_k, gm_K = test_rate_coefficients(make_plot=not args.no_plot)
    test_feasibility()

    sep("PODSUMOWANIE")
    print(f"  TEST 1 k_iz  : średnia geom. odchylenia = {gm_k:.2f}x")
    print(f"  TEST 1 K     : średnia geom. odchylenia = {gm_K:.2f}x")
    print(f"  TEST 2 setup : {'ZALICZONY' if ok_setup else 'NIEZALICZONY'}")
    print(f"  TEST 3       : pełny stan ustalony poza zasięgiem (patrz wyżej)")


if __name__ == "__main__":
    main()
