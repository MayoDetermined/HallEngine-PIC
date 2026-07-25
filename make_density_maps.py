"""Mapy uśrednionej gęstości plazmy z naniesioną wiązką RE i warstwami sheath.

    python make_density_maps.py              # wszystkie mapy
    python make_density_maps.py --only 1d
    python make_density_maps.py --only 2d

Produkuje:
    map_1d3v.png            — mapa CZASOPRZESTRZENNA (x, t) + wiązka RE + sheath
    map_2d3v_zr.png         — mapa przestrzenna 2D (z-r)
    map_2d3v_ztheta.png     — mapa przestrzenna 2D (z-θ)

DEFINICJE (żeby mapy były jednoznaczne)
---------------------------------------
gęstość plazmy   : n_pl = (n_e + n_i)/2 — w obszarze quasineutralnym równa n_e≈n_i,
                   a w warstwie sheath jest po prostu miarą lokalnego zagęszczenia.
wiązka RE        : n_RE = gęstość elektronów o energii > E_RE, deponowana tym
                   samym jądrem CIC co pełna gęstość (spójna normalizacja wag).
warstwa sheath   : obszar ZŁAMANIA QUASINEUTRALNOŚCI, mierzony względnym
                   rozdzieleniem ładunku
                        s(x) = (n_i − n_e) / max(n_pl)
                   Sheath to |s| > SHEATH_THR. To definicja fizyczna, nie
                   arbitralny pasek przy brzegu — warstwa sama się lokalizuje
                   przy anodzie/katodzie (i przy ściankach w geometrii z-r).

Uśrednianie pomija początkowy transient (SKIP_FRAC długości przebiegu).
"""

import argparse
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SHEATH_THR = 0.10      # próg rozdzielenia ładunku definiujący warstwę
SKIP_FRAC = 0.30       # pomijany transient początkowy


# ---------------------------------------------------------------- 1D
def deposit_1d(x, w, cfg):
    """CIC 1. rzędu dla dowolnego podzbioru cząstek -> gęstość [1/m³]."""
    n = np.zeros(cfg.n_nodes)
    if x.size == 0:
        return n
    fi = x / cfg.dx
    i = np.clip(np.floor(fi).astype(np.int64), 0, cfg.Nx - 1)
    f = fi - i
    c = w / cfg.dx
    np.add.at(n, i, c * (1.0 - f))
    np.add.at(n, i + 1, c * f)
    n[0] *= 2.0
    n[-1] *= 2.0
    return n


def run_1d(n_steps=16000, n_frames=160):
    from hall_pic import Config, Simulation
    cfg = Config()
    cfg.live_view = False
    cfg.__post_init__()
    sim = Simulation(cfg)

    every = max(1, n_steps // n_frames)
    skip = int(SKIP_FRAC * n_steps)
    st_pl, st_re, st_s, times = [], [], [], []
    acc_e = np.zeros(cfg.n_nodes); acc_i = np.zeros(cfg.n_nodes)
    acc_re = np.zeros(cfg.n_nodes); n_acc = 0

    t0 = time.time()
    for k in range(1, n_steps + 1):
        sim.step_once()
        if k % every:
            continue
        el, io = sim.electrons, sim.ions
        eps = el.kinetic_energy_eV()
        ne = deposit_1d(el.ax, el.aw, cfg)
        ni = deposit_1d(io.ax, io.aw, cfg)
        m = eps > cfg.E_RE_eV
        nre = deposit_1d(el.ax[m], el.aw[m], cfg)
        npl = 0.5 * (ne + ni)
        ref = max(npl.max(), 1e-30)
        st_pl.append(npl); st_re.append(nre); st_s.append((ni - ne) / ref)
        times.append(sim.t)
        if k > skip:
            acc_e += ne; acc_i += ni; acc_re += nre; n_acc += 1
    print(f"   1D3V: {n_steps} kroków, {sim.t*1e9:.0f} ns, {time.time()-t0:.0f} s")
    return (cfg, np.array(times), np.array(st_pl), np.array(st_re), np.array(st_s),
            acc_e / n_acc, acc_i / n_acc, acc_re / n_acc)


def _smooth(a, size):
    """Wygładzenie boxcar. Kontury liczone na CHWILOWYCH klatkach PIC są
    zdominowane przez szum statystyczny — próg sheath byłby przekraczany
    losowo w całej objętości. Wygładzamy WYŁĄCZNIE pola do konturowania;
    mapa gęstości i profile uśrednione pozostają surowe."""
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(a, size=size, mode="nearest")
    except Exception:
        return a


def fig_1d(cfg, t, st_pl, st_re, st_s, ne, ni, nre, out="map_1d3v.png"):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6),
                           gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
    fig.suptitle("PIC 1D3V — mapa czasoprzestrzenna gęstości plazmy "
                 "z wiązką RE i warstwami sheath", fontsize=13, weight="bold")
    mm, ns = 1e3, 1e9
    ext = [0, cfg.L * mm, t[0] * ns, t[-1] * ns]

    # ---- (a) mapa czasoprzestrzenna + kontury RE + sheath ----
    a = ax[0]
    im = a.imshow(st_pl, origin="lower", aspect="auto", extent=ext, cmap="viridis")
    plt.colorbar(im, ax=a, label="gęstość plazmy [m⁻³]", fraction=0.046)
    X = np.linspace(0, cfg.L * mm, cfg.n_nodes)
    T = t * ns
    # wiązka RE — kontury na polu wygładzonym (patrz _smooth)
    re_sm = _smooth(st_re, (5, 7))
    if re_sm.max() > 0:
        lv = re_sm.max() * np.array([0.25, 0.5, 0.75])
        a.contour(X, T, re_sm, levels=lv, colors="#00e5ff", linewidths=1.3)
    # sheath — obszary złamania quasineutralności
    s_sm = _smooth(np.abs(st_s), (5, 7))
    a.contour(X, T, s_sm, levels=[SHEATH_THR], colors="red", linewidths=1.6)
    a.set_title("(a) ⟨n_plazmy⟩(x,t)  +  wiązka RE (cyjan)  +  sheath (czerwony)")
    a.set_xlabel("x [mm]"); a.set_ylabel("t [ns]")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color="#00e5ff", lw=1.6, label="wiązka RE"),
                      Line2D([], [], color="red", lw=1.8,
                             label=f"sheath |Δn|/n > {SHEATH_THR:.0%}")],
             fontsize=8, loc="upper right", framealpha=0.85)

    # ---- (b) profile uśrednione po czasie ----
    a = ax[1]
    a.plot(X, ne, "b-", lw=1.5, label="⟨n_e⟩")
    a.plot(X, ni, "r-", lw=1.5, label="⟨n_i⟩")
    a.set_xlabel("x [mm]"); a.set_ylabel("n [m⁻³]"); a.grid(alpha=0.3)
    a.set_title("(b) Profile uśrednione + gęstość RE")
    a2 = a.twinx()
    a2.plot(X, nre, color="#00b3cc", lw=1.8, label="⟨n_RE⟩")
    a2.set_ylabel("n_RE [m⁻³]", color="#00b3cc")
    a2.tick_params(axis="y", colors="#00b3cc")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center")

    # ---- (c) rozdzielenie ładunku = lokalizacja warstw ----
    a = ax[2]
    npl = 0.5 * (ne + ni)
    s = (ni - ne) / max(npl.max(), 1e-30)
    a.plot(X, s, "k-", lw=1.6)
    a.axhline(0, color="gray", lw=0.7)
    a.axhline(SHEATH_THR, color="red", ls="--", lw=1.2)
    a.axhline(-SHEATH_THR, color="red", ls="--", lw=1.2)
    a.fill_between(X, s, SHEATH_THR, where=(s > SHEATH_THR),
                   color="red", alpha=0.25, label="sheath (nadmiar jonów)")
    a.fill_between(X, s, -SHEATH_THR, where=(s < -SHEATH_THR),
                   color="blue", alpha=0.25, label="nadmiar elektronów")
    a.set_title("(c) Rozdzielenie ładunku (n_i−n_e)/n")
    a.set_xlabel("x [mm]"); a.set_ylabel("względne rozdzielenie")
    a.grid(alpha=0.3); a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=115)
    print(f"   zapisano: {out}")


# ---------------------------------------------------------------- 2D
def deposit_2d(x1, x2, w, cfg):
    """CIC dwuliniowa dla dowolnego podzbioru cząstek -> gęstość [1/m³]."""
    nt = cfg.n1_nodes * cfg.n2_nodes
    if x1.size == 0:
        return np.zeros((cfg.n1_nodes, cfg.n2_nodes))
    f1 = x1 / cfg.h1
    i = np.clip(np.floor(f1).astype(np.int64), 0, cfg.N1 - 1)
    a = f1 - i
    f2 = x2 / cfg.h2
    j = np.floor(f2).astype(np.int64)
    b = f2 - j
    if cfg.x2_bc == "periodic":
        j = np.mod(j, cfg.N2); j2 = np.mod(j + 1, cfg.N2)
    else:
        j = np.clip(j, 0, cfg.N2 - 1); j2 = j + 1
    n2n = cfg.n2_nodes
    vals = w / (cfg.h1 * cfg.h2)
    out = np.zeros(nt)
    for idx, wt in ((i * n2n + j, (1 - a) * (1 - b)),
                    ((i + 1) * n2n + j, a * (1 - b)),
                    (i * n2n + j2, (1 - a) * b),
                    ((i + 1) * n2n + j2, a * b)):
        out += np.bincount(idx, weights=vals * wt, minlength=nt)
    out = out.reshape(cfg.n1_nodes, n2n)
    out[0, :] *= 2.0; out[-1, :] *= 2.0
    if cfg.x2_bc != "periodic":
        out[:, 0] *= 2.0; out[:, -1] *= 2.0
    return out


def run_2d(geometry, n_steps=4000, sample_every=25):
    from hall_pic2d import Config2D, Simulation2D
    cfg = Config2D()
    cfg.geometry = geometry
    cfg.N1, cfg.N2, cfg.n_ppc = 96, 48, 12
    cfg.live_view = False
    cfg.__post_init__()
    sim = Simulation2D(cfg)

    skip = int(SKIP_FRAC * n_steps)
    shape = (cfg.n1_nodes, cfg.n2_nodes)
    acc_e = np.zeros(shape); acc_i = np.zeros(shape); acc_re = np.zeros(shape)
    n_acc = 0
    t0 = time.time()
    for k in range(1, n_steps + 1):
        sim.step_once()
        if k % sample_every or k <= skip:
            continue
        el, io = sim.electrons, sim.ions
        eps = el.kinetic_energy_eV()
        m = eps > cfg.E_RE_eV
        acc_e += deposit_2d(el.ax1, el.ax2, el.aw, cfg)
        acc_i += deposit_2d(io.ax1, io.ax2, io.aw, cfg)
        acc_re += deposit_2d(el.ax1[m], el.ax2[m], el.aw[m], cfg)
        n_acc += 1
    print(f"   2D3V {geometry}: {n_steps} kroków, {sim.t*1e9:.1f} ns, "
          f"{time.time()-t0:.0f} s ({n_acc} próbek)")
    return cfg, acc_e / n_acc, acc_i / n_acc, acc_re / n_acc


def fig_2d(cfg, ne, ni, nre, out):
    npl = 0.5 * (ne + ni)
    ref = max(npl.max(), 1e-30)
    s = (ni - ne) / ref

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    nice = "osiowo-promieniowa (z-r)" if cfg.geometry == "z-r" else "osiowo-azymutalna (z-θ)"
    fig.suptitle(f"PIC 2D3V ({nice}) — uśredniona gęstość plazmy "
                 "z wiązką RE i warstwami sheath", fontsize=13, weight="bold")
    mm = 1e3
    ext = [0, cfg.L2 * mm, 0, cfg.L1 * mm]
    X = np.linspace(0, cfg.L2 * mm, cfg.n2_nodes)
    Y = np.linspace(0, cfg.L1 * mm, cfg.n1_nodes)

    # (a) gęstość + kontury RE + sheath
    a = ax[0]
    im = a.imshow(npl, origin="lower", aspect="auto", extent=ext, cmap="viridis")
    plt.colorbar(im, ax=a, label="⟨n_plazmy⟩ [m⁻³]", fraction=0.046)
    # kontury na polach lekko wygładzonych — przy 12 cząstkach/komórkę
    # statystyka na węzeł jest zbyt szumiąca, by konturować ją surowo
    nre_sm = _smooth(nre, (3, 3))
    s_sm = _smooth(np.abs(s), (3, 3))
    if nre_sm.max() > 0:
        a.contour(X, Y, nre_sm, levels=nre_sm.max() * np.array([0.25, 0.5, 0.75]),
                  colors="#00e5ff", linewidths=1.3)
    a.contour(X, Y, s_sm, levels=[SHEATH_THR], colors="red", linewidths=1.6)
    a.set_title("(a) ⟨n_plazmy⟩ + wiązka RE (cyjan) + sheath (czerwony)")
    a.set_xlabel(cfg.x2_label); a.set_ylabel("z [mm]")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color="#00e5ff", lw=1.6, label="wiązka RE"),
                      Line2D([], [], color="red", lw=1.8,
                             label=f"sheath |Δn|/n > {SHEATH_THR:.0%}")],
             fontsize=8, loc="upper right", framealpha=0.85)

    # (b) sama gęstość RE
    a = ax[1]
    im2 = a.imshow(nre, origin="lower", aspect="auto", extent=ext, cmap="inferno")
    plt.colorbar(im2, ax=a, label="⟨n_RE⟩ [m⁻³]", fraction=0.046)
    a.set_title("(b) Gęstość elektronów RE (E > próg)")
    a.set_xlabel(cfg.x2_label); a.set_ylabel("z [mm]")

    # (c) rozdzielenie ładunku
    a = ax[2]
    lim = max(np.abs(s).max(), 1e-6)
    im3 = a.imshow(s, origin="lower", aspect="auto", extent=ext, cmap="coolwarm",
                   vmin=-lim, vmax=lim)
    plt.colorbar(im3, ax=a, label="(n_i − n_e)/n", fraction=0.046)
    a.contour(X, Y, s_sm, levels=[SHEATH_THR], colors="k", linewidths=1.2)
    a.set_title("(c) Rozdzielenie ładunku — lokalizacja warstw")
    a.set_xlabel(cfg.x2_label); a.set_ylabel("z [mm]")

    fig.tight_layout()
    fig.savefig(out, dpi=115)
    print(f"   zapisano: {out}")


# ----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["1d", "2d"], default=None)
    args = p.parse_args()
    print(f"Mapy gęstości (transient pominięty: {SKIP_FRAC:.0%}, "
          f"próg sheath: {SHEATH_THR:.0%})")
    if args.only != "2d":
        print(" [1D3V] ...")
        fig_1d(*run_1d())
    if args.only != "1d":
        for g, fn in (("z-r", "map_2d3v_zr.png"), ("z-theta", "map_2d3v_ztheta.png")):
            print(f" [2D3V {g}] ...")
            cfg, ne, ni, nre = run_2d(g)
            fig_2d(cfg, ne, ni, nre, fn)
    print("Gotowe.")


if __name__ == "__main__":
    main()
