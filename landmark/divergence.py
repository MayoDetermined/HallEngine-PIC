"""Rozjazd między domknięciem LANDMARK (fluid) a kodami kinetycznymi 1D3V/2D3V.

    python -m landmark.divergence

CO JEST PORÓWNYWANE I DLACZEGO
------------------------------
Model płynowy LANDMARK czerpie jonizację z tabeli k_iz(ε) zbudowanej przy
założeniu, że EEDF jest MAXWELLOWSKI o średniej energii ε. To założenie jest
wprasowane w benchmark — nie da się go z niego wyjąć.

Kody PIC liczą rozkład RZECZYWISTY, który przy formowaniu wiązek RE jest silnie
niemaxwellowski (nadmiar w ogonie wysokoenergetycznym). Ponieważ przekrój
jonizacyjny rośnie daleko powyżej progu 12.13 eV, ogon waży w k_iz nieproporcjonalnie
mocno. Rozjazd mierzymy wprost:

    k_iz^kinetyczne = Σ_p w_p σ_iz(E_p) v_p / Σ_p w_p     (rozkład z PIC)
    k_iz^LANDMARK   = tabela(ε̄),  ε̄ = Σ_p w_p E_p / Σ_p w_p

Iloraz tych dwóch wielkości to BŁĄD, jaki popełnia domknięcie płynowe przy tej
samej średniej energii. Nie wymaga to zbieżnego rozwiązania modelu płynowego —
porównujemy samo domknięcie, punkt po punkcie, na danych referencyjnych LANDMARK.
"""

import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON
from hall_pic import cross_sections as xs
from . import rates


def kinetic_k_iz(energies_eV, weights):
    """⟨σ_iz·v⟩ po RZECZYWISTYM rozkładzie cząstek."""
    if energies_eV.size == 0 or weights.sum() <= 0:
        return 0.0, 0.0
    v = np.sqrt(2.0 * energies_eV * E_CHARGE / M_ELECTRON)
    sig = xs.sigma_ionization_e(energies_eV)
    k = np.sum(weights * sig * v) / np.sum(weights)
    eps_mean = np.sum(weights * energies_eV) / np.sum(weights)
    return float(k), float(eps_mean)


def maxwellian_k_iz(eps_mean_eV, eps_tab, kiz_tab):
    """k_iz jakie przyjąłby LANDMARK przy tej samej średniej energii."""
    return float(np.interp(eps_mean_eV, eps_tab, kiz_tab))


# ----------------------------------------------------------------------
def sample_1d(n_steps=9000, every=500):
    from hall_pic import Config, Simulation
    cfg = Config()
    cfg.live_view = False
    cfg.__post_init__()
    sim = Simulation(cfg)
    rec = []
    for i in range(n_steps):
        sim.step_once()
        if (i + 1) % every == 0:
            el = sim.electrons
            e = el.kinetic_energy_eV()
            w = el.aw.copy()
            k, em = kinetic_k_iz(e, w)
            rec.append((sim.t, em, k))
    el = sim.electrons
    return rec, el.kinetic_energy_eV(), el.aw.copy()


def sample_2d(geometry="z-r", n_steps=2500, every=250):
    from hall_pic2d import Config2D, Simulation2D
    cfg = Config2D()
    cfg.geometry = geometry
    cfg.N1, cfg.N2, cfg.n_ppc = 96, 48, 12
    cfg.live_view = False
    cfg.__post_init__()
    sim = Simulation2D(cfg)
    rec = []
    for i in range(n_steps):
        sim.step_once()
        if (i + 1) % every == 0:
            el = sim.electrons
            e = el.kinetic_energy_eV()
            w = el.aw.copy()
            k, em = kinetic_k_iz(e, w)
            rec.append((sim.t, em, k))
    el = sim.electrons
    return rec, el.kinetic_energy_eV(), el.aw.copy()


# ----------------------------------------------------------------------
def maxwellian_eedf(E, eps_mean):
    """Znormalizowany EEDF maxwellowski o tej samej średniej energii (ε̄ = 3T/2)."""
    T = 2.0 / 3.0 * max(eps_mean, 1e-3)
    return (2.0 / np.sqrt(np.pi)) * T**-1.5 * np.sqrt(np.maximum(E, 0)) * np.exp(-E / T)


def main():
    eps_tab, kiz_tab, _ = rates.load_landmark_tables()

    print("=" * 72)
    print(" Rozjazd domknięcia: LANDMARK (maxwellian) vs PIC (rozkład rzeczywisty)")
    print("=" * 72)
    print(" [1/3] próbkowanie 1D3V ...")
    rec1, e1, w1 = sample_1d()
    print(" [2/3] próbkowanie 2D3V (z-r) ...")
    rec2, e2, w2 = sample_2d("z-r")
    print(" [3/3] próbkowanie 2D3V (z-theta) ...")
    rec3, e3, w3 = sample_2d("z-theta")

    print(f"\n {'kod':12s} {'t [ns]':>8} {'ε̄ [eV]':>9} {'k_iz kin.':>12}"
          f" {'k_iz LM':>12} {'kin/LM':>8}")
    rows = []
    for label, rec in (("1D3V", rec1), ("2D3V z-r", rec2), ("2D3V z-θ", rec3)):
        for (t, em, k) in rec:
            km = maxwellian_k_iz(em, eps_tab, kiz_tab)
            r = k / km if km > 0 else np.nan
            rows.append((label, t, em, k, km, r))
        t, em, k = rec[-1]
        km = maxwellian_k_iz(em, eps_tab, kiz_tab)
        print(f" {label:12s} {t*1e9:8.1f} {em:9.2f} {k:12.4e} {km:12.4e}"
              f" {k/km if km>0 else float('nan'):8.2f}")

    _plot(rows, (e1, w1), (e2, w2), (e3, w3), eps_tab, kiz_tab)


def _plot(rows, d1, d2, d3, eps_tab, kiz_tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Rozjazd: domknięcie maxwellowskie LANDMARK vs kinetyczne PIC",
                 fontsize=13)

    # --- panel 1-2: EEDF PIC vs maxwellian o tej samej średniej ---
    for a, (e, w), label, col in ((ax[0, 0], d1, "1D3V", "tab:blue"),
                                  (ax[0, 1], d2, "2D3V (z-r)", "tab:red")):
        em = np.sum(w * e) / np.sum(w)
        bins = np.linspace(0, max(np.percentile(e, 99.9), 60), 90)
        ctr = 0.5 * (bins[1:] + bins[:-1])
        h, _ = np.histogram(e, bins=bins, weights=w, density=True)
        a.semilogy(ctr, np.maximum(h, 1e-12), color=col, lw=1.8,
                   label=f"PIC {label} (rzeczywisty)")
        mx = maxwellian_eedf(ctr, em)
        a.semilogy(ctr, np.maximum(mx, 1e-12), "k--", lw=1.6,
                   label=f"maxwellian (ε̄={em:.1f} eV)\nzałożenie LANDMARK")
        a.set_title(f"EEDF — {label}")
        a.set_xlabel("energia [eV]"); a.set_ylabel("f(E) [1/eV]")
        a.set_ylim(1e-8, None); a.grid(alpha=0.3); a.legend(fontsize=8)
        a.axvline(12.13, color="green", ls=":", lw=1.2)
        a.text(12.5, a.get_ylim()[0]*30, "próg jonizacji", fontsize=7,
               color="green", rotation=90)

    # --- panel 3: k_iz kinetyczne vs tabela LANDMARK ---
    a = ax[1, 0]
    a.semilogy(eps_tab, np.maximum(kiz_tab, 1e-25), "k-", lw=2,
               label="LANDMARK k_iz(ε̄) — maxwellian")
    style = {"1D3V": ("tab:blue", "o"), "2D3V z-r": ("tab:red", "s"),
             "2D3V z-θ": ("tab:green", "^")}
    for label, (col, mk) in style.items():
        pts = [(r[2], r[3]) for r in rows if r[0] == label]
        if pts:
            xs_, ys_ = zip(*pts)
            a.semilogy(xs_, np.maximum(ys_, 1e-25), mk, color=col, ms=6,
                       alpha=0.8, label=f"PIC {label} (kinetyczne)")
    a.set_xlabel("średnia energia elektronów ε̄ [eV]")
    a.set_ylabel("k_iz [m³/s]")
    a.set_title("Współczynnik jonizacji przy tej samej ε̄")
    a.grid(alpha=0.3); a.legend(fontsize=8)

    # --- panel 4: iloraz = błąd domknięcia płynowego ---
    a = ax[1, 1]
    for label, (col, mk) in style.items():
        pts = [(r[2], r[5]) for r in rows if r[0] == label and np.isfinite(r[5])]
        if pts:
            xs_, ys_ = zip(*pts)
            a.semilogy(xs_, ys_, mk, color=col, ms=7, alpha=0.85, label=f"PIC {label}")
    a.axhline(1.0, color="k", ls="--", lw=1.5)
    a.axhspan(0.5, 2.0, color="green", alpha=0.12, label="±czynnik 2")
    a.set_xlabel("średnia energia elektronów ε̄ [eV]")
    a.set_ylabel("k_iz kinetyczne / k_iz maxwellowskie")
    a.set_title("BŁĄD domknięcia maxwellowskiego LANDMARK")
    a.grid(alpha=0.3); a.legend(fontsize=8)

    fig.tight_layout()
    out = "landmark_divergence.png"
    fig.savefig(out, dpi=110)
    print(f"\n wykres zapisany: {out}")


if __name__ == "__main__":
    main()
