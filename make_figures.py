"""Gotowe rysunki podsumowujące przebiegi, osobno dla obu wersji symulacji.

    python make_figures.py                 # wszystkie trzy rysunki
    python make_figures.py --only 1d       # tylko wersja jednowymiarowa
    python make_figures.py --only 2d       # tylko obie płaszczyzny wersji dwuwymiarowej

Powstają trzy pliki obrazów: jeden dla wersji jednowymiarowej i dwa dla wersji
na płaszczyźnie, po jednym na każdą płaszczyznę kanału. W odróżnieniu od
podglądu na żywo te rysunki pokazują, jak sytuacja zmienia się w czasie, bo
wiązka nie powstaje od razu, tylko stopniowo się formuje. Dlatego rozkład
elektronów i rozkład energii pokazujemy w kilku chwilach.
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

from hall_pic.constants import E_CHARGE

PS_CMAP = "inferno"


# Wersja jednowymiarowa.
def run_1d(n_steps=16000, n_snap=3):
    """Przeprowadza przebieg jednowymiarowy i zbiera migawki oraz przebieg obwodu."""
    from hall_pic import Config, Simulation
    from hall_pic import pusher

    cfg = Config()
    cfg.live_view = False
    cfg.__post_init__()
    sim = Simulation(cfg)

    snap_at = np.linspace(n_steps // 4, n_steps, n_snap).astype(int)
    snaps, hist = [], []

    t0 = time.time()
    for i in range(1, n_steps + 1):
        sim.step_once()
        if i % 40 == 0:
            el = sim.electrons
            eps = el.kinetic_energy_eV()
            hist.append((sim.t, sim.I_d, sim.circuit.I_L, sim.circuit.V_C, sim.I_RE,
                         int(np.sum(eps > cfg.E_RE_eV)),
                         float(eps.max()) if el.N else 0.0,
                         float(np.average(eps, weights=el.aw)) if el.N else 0.0))
        if i in snap_at:
            el = sim.electrons
            st = max(1, el.N // 6000)
            snaps.append(dict(
                t=sim.t, x=el.ax[::st].copy(), vx=el.avx[::st].copy(),
                eps_s=el.kinetic_energy_eV()[::st].copy(),
                eps_all=el.kinetic_energy_eV().copy(), w_all=el.aw.copy()))
    print(f"   1D3V: {n_steps} kroków, {sim.t*1e9:.0f} ns, {time.time()-t0:.0f} s")

    ne = pusher.deposit_number_density(sim.electrons, cfg)
    ni = pusher.deposit_number_density(sim.ions, cfg)
    return cfg, sim, snaps, np.array(hist), ne, ni


def fig_1d(cfg, sim, snaps, hist, ne, ni, out="fig_1d3v.png"):
    """Składa sześciopanelowy rysunek podsumowujący przebieg jednowymiarowy."""
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("PIC 1D3V  silnik Halla SPT-100: formowanie wiązki elektronów RE",
                 fontsize=14, weight="bold")
    mm = 1e3
    vmax = max(cfg.E_RE_eV * 4, 150)

    # Panel pierwszy: rozkład elektronów w ostatniej chwili.
    s = snaps[-1]
    a = ax[0, 0]
    sc = a.scatter(s["x"] * mm, s["vx"] / 1e6, c=s["eps_s"], s=3, cmap=PS_CMAP,
                   vmin=0, vmax=vmax, alpha=0.65)
    a.axhline(0, color="gray", lw=0.6)
    a.set_title(f"(a) Przestrzeń fazowa e⁻   t = {s['t']*1e9:.0f} ns")
    a.set_xlabel("x [mm]"); a.set_ylabel("v_x [10⁶ m/s]"); a.grid(alpha=0.3)
    plt.colorbar(sc, ax=a, label="energia [eV]", fraction=0.046)

    # Panel drugi: rozkład energii w kilku chwilach.
    a = ax[0, 1]
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(snaps)))
    hmax = 0.0
    # Oś energii sięga do najwyższej energii, jaka wystąpiła w którejkolwiek
    # migawce, żeby prawy kraniec wykresu pokrywał się z odczytem najwyższej
    # energii w panelu obok. Wcześniej oś urywaliśmy na wysokim centylu, przez
    # co najwyższa energia leżała daleko poza widoczną częścią wykresu.
    e_top = max(sn["eps_all"].max() for sn in snaps) * 1.02
    for k, sn in enumerate(snaps):
        bins = np.linspace(0, e_top, 70)
        h, e = np.histogram(sn["eps_all"], bins=bins, weights=sn["w_all"])
        h = h.astype(float)
        # Puste przedziały zamieniamy na wartość pustą, żeby ich nie rysować.
        # Podciąganie ich do zera dawało na skali logarytmicznej pionowe kreski
        # spadające na sam dół zamiast czytelnego ogona.
        h[h <= 0] = np.nan
        hmax = max(hmax, np.nanmax(h))
        a.semilogy(0.5 * (e[1:] + e[:-1]), h, color=cols[k], lw=1.7,
                   marker="o", ms=2.5, label=f"t = {sn['t']*1e9:.0f} ns")
    a.axvline(cfg.E_RE_eV, color="red", ls="--", lw=1.4,
              label=f"próg RE = {cfg.E_RE_eV:.0f} eV")
    a.set_ylim(hmax / 1e7, hmax * 3)
    a.set_title("(b) EEDF  narastanie ogona RE")
    a.set_xlabel("energia [eV]"); a.set_ylabel("waga (log)")
    a.grid(alpha=0.3); a.legend(fontsize=8)

    # Panel trzeci: gęstości obu gatunków i kształt pola magnetycznego.
    a = ax[0, 2]
    a.plot(cfg.x_nodes * mm, ne, "b-", lw=1.3, label="n_e")
    a.plot(cfg.x_nodes * mm, ni, "r-", lw=1.3, label="n_i")
    a.set_title("(c) Gęstości i pole magnetyczne")
    a.set_xlabel("x [mm]"); a.set_ylabel("n [m⁻³]")
    a.grid(alpha=0.3); a.legend(loc="lower center", fontsize=8)
    a2 = a.twinx()
    a2.plot(cfg.x_nodes * mm, cfg.B_profile(cfg.x_nodes) * 1e4, "g--", lw=1.4, alpha=0.7)
    a2.set_ylabel("B [G]", color="g"); a2.tick_params(axis="y", colors="g")

    # Panel czwarty: potencjał i pole elektryczne.
    a = ax[1, 0]
    a.plot(cfg.x_nodes * mm, sim.phi, "m-", lw=1.8)
    a.set_title("(d) Potencjał i pole elektryczne")
    a.set_xlabel("x [mm]"); a.set_ylabel("φ [V]", color="m")
    a.tick_params(axis="y", colors="m"); a.grid(alpha=0.3)
    a3 = a.twinx()
    a3.plot(cfg.x_nodes * mm, sim.E / 1e3, "c-", lw=1.2, alpha=0.75)
    a3.set_ylabel("E_x [kV/m]", color="c"); a3.tick_params(axis="y", colors="c")

    # Panel piąty: przebieg prądów i napięcia w obwodzie.
    a = ax[1, 1]
    t_ns = hist[:, 0] * 1e9
    a.plot(t_ns, hist[:, 1], "b-", lw=1, alpha=0.8, label="I_d (wyładowanie)")
    a.plot(t_ns, hist[:, 2], "r-", lw=1.6, label="I_L (obwód)")
    a.plot(t_ns, hist[:, 4], color="orange", lw=1.6, label="I_RE (wiązka)")
    a.set_title("(e) Obwód RLC (cel 4 A)")
    a.set_xlabel("t [ns]"); a.set_ylabel("I [A]")
    a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left")
    a4 = a.twinx()
    a4.plot(t_ns, hist[:, 3], "k--", lw=1.2, alpha=0.6)
    a4.set_ylabel("V_C [V]")

    # Panel szósty: liczba i energia rozpędzonych elektronów w czasie.
    a = ax[1, 2]
    a.plot(t_ns, hist[:, 5], color="darkorange", lw=1.6, label="liczba e⁻ RE")
    a.set_title("(f) Populacja RE i energia")
    a.set_xlabel("t [ns]"); a.set_ylabel("N(E > próg RE)", color="darkorange")
    a.tick_params(axis="y", colors="darkorange"); a.grid(alpha=0.3)
    a5 = a.twinx()
    a5.plot(t_ns, hist[:, 6], "purple", lw=1.3, label="max E")
    a5.plot(t_ns, hist[:, 7], "gray", lw=1.3, ls="--", label="⟨E⟩")
    a5.set_ylabel("energia [eV]")
    a5.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=115)
    print(f"   zapisano: {out}")


# Wersja na płaszczyźnie.
def run_2d(geometry, n_steps=4000):
    """Przeprowadza przebieg na płaszczyźnie i zbiera migawki oraz przebieg obwodu."""
    from hall_pic2d import Config2D, Simulation2D
    from hall_pic2d import pusher2d

    cfg = Config2D()
    cfg.geometry = geometry
    cfg.N1, cfg.N2, cfg.n_ppc = 96, 48, 12
    cfg.live_view = False
    cfg.__post_init__()
    sim = Simulation2D(cfg)

    hist, snaps = [], []
    snap_at = {n_steps // 3, 2 * n_steps // 3, n_steps}
    t0 = time.time()
    for i in range(1, n_steps + 1):
        sim.step_once()
        if i % 25 == 0:
            el = sim.electrons
            eps = el.kinetic_energy_eV()
            hist.append((sim.t, sim.I_d, sim.circuit.I_L, sim.circuit.V_C, sim.I_RE,
                         int(np.sum(eps > cfg.E_RE_eV)),
                         float(eps.max()) if el.N else 0.0, el.N))
        if i in snap_at:
            el = sim.electrons
            snaps.append(dict(t=sim.t, eps_all=el.kinetic_energy_eV().copy(),
                              w_all=el.aw.copy()))
    print(f"   2D3V {geometry}: {n_steps} kroków, {sim.t*1e9:.1f} ns, {time.time()-t0:.0f} s")
    ne = pusher2d.deposit_number_density(sim.electrons, cfg)
    return cfg, sim, snaps, np.array(hist), ne


def fig_2d(cfg, sim, snaps, hist, ne, out):
    """Składa sześciopanelowy rysunek podsumowujący przebieg na płaszczyźnie."""
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    nice = "osiowo-promieniowa (z-r)" if cfg.geometry == "z-r" else "osiowo-azymutalna (z-θ)"
    fig.suptitle(f"PIC 2D3V  silnik Halla, płaszczyzna {nice}",
                 fontsize=14, weight="bold")
    mm = 1e3
    extent = [0.0, cfg.L2 * mm, 0.0, cfg.L1 * mm]
    el = sim.electrons

    # Panel pierwszy: mapa gęstości elektronów.
    a = ax[0, 0]
    im = a.imshow(ne, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    a.set_title(f"(a) Gęstość elektronów   t = {sim.t*1e9:.1f} ns")
    a.set_xlabel(cfg.x2_label); a.set_ylabel("z [mm]")
    plt.colorbar(im, ax=a, label="n_e [m⁻³]", fraction=0.046)

    # Panel drugi: mapa potencjału z liniami stałej wartości.
    a = ax[0, 1]
    im2 = a.imshow(sim.phi, origin="lower", aspect="auto", extent=extent, cmap="plasma")
    try:
        a.contour(np.linspace(0, cfg.L2 * mm, cfg.n2_nodes),
                  np.linspace(0, cfg.L1 * mm, cfg.n1_nodes),
                  sim.phi, levels=9, colors="w", linewidths=0.45, alpha=0.65)
    except Exception:
        pass
    a.set_title("(b) Potencjał elektryczny")
    a.set_xlabel(cfg.x2_label); a.set_ylabel("z [mm]")
    plt.colorbar(im2, ax=a, label="φ [V]", fraction=0.046)

    # Panel trzeci: rozkład elektronów w przestrzeni położeń i prędkości.
    a = ax[0, 2]
    st = max(1, el.N // 6000)
    eps_s = el.kinetic_energy_eV()[::st]
    sc = a.scatter(el.ax1[::st] * mm, el.av1[::st] / 1e6, c=eps_s, s=3,
                   cmap=PS_CMAP, vmin=0, vmax=max(cfg.E_RE_eV * 4, 150), alpha=0.65)
    a.axhline(0, color="gray", lw=0.6)
    a.set_title("(c) Przestrzeń fazowa e⁻ (z, v_z)")
    a.set_xlabel("z [mm]"); a.set_ylabel("v_z [10⁶ m/s]"); a.grid(alpha=0.3)
    plt.colorbar(sc, ax=a, label="energia [eV]", fraction=0.046)

    # Panel czwarty: rozkład energii w kilku chwilach.
    a = ax[1, 0]
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(snaps)))
    # Oś energii sięga do najwyższej energii z migawek, tak jak w wersji
    # jednowymiarowej, żeby zgadzała się z odczytem najwyższej energii.
    e_top = max(sn["eps_all"].max() for sn in snaps) * 1.02
    hmax = 0.0
    for k, sn in enumerate(snaps):
        bins = np.linspace(0, e_top, 70)
        h, e = np.histogram(sn["eps_all"], bins=bins, weights=sn["w_all"])
        h = h.astype(float)
        h[h <= 0] = np.nan          # puste przedziały pomijamy, tak jak w wersji jednowymiarowej
        hmax = max(hmax, np.nanmax(h))
        a.semilogy(0.5 * (e[1:] + e[:-1]), h, color=cols[k], lw=1.7,
                   label=f"t = {sn['t']*1e9:.1f} ns")
    a.axvline(cfg.E_RE_eV, color="red", ls="--", lw=1.4, label="próg RE")
    a.set_ylim(hmax / 1e6, hmax * 3)
    a.set_title("(d) EEDF  narastanie ogona RE")
    a.set_xlabel("energia [eV]"); a.set_ylabel("waga (log)")
    a.grid(alpha=0.3); a.legend(fontsize=8)

    # Panel piąty: przebieg prądów i napięcia w obwodzie.
    a = ax[1, 1]
    t_ns = hist[:, 0] * 1e9
    a.plot(t_ns, hist[:, 1], "b-", lw=1, alpha=0.8, label="I_d")
    a.plot(t_ns, hist[:, 2], "r-", lw=1.6, label="I_L")
    a.plot(t_ns, hist[:, 4], color="orange", lw=1.6, label="I_RE")
    a.set_title("(e) Obwód RLC")
    a.set_xlabel("t [ns]"); a.set_ylabel("I [A]")
    a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left")
    a4 = a.twinx()
    a4.plot(t_ns, hist[:, 3], "k--", lw=1.2, alpha=0.6)
    a4.set_ylabel("V_C [V]")

    # Panel szósty: liczba rozpędzonych elektronów, energia i liczba czastek.
    a = ax[1, 2]
    a.plot(t_ns, hist[:, 5], color="darkorange", lw=1.6)
    a.set_title("(f) Populacja RE, energia i APR")
    a.set_xlabel("t [ns]"); a.set_ylabel("N(E > próg RE)", color="darkorange")
    a.tick_params(axis="y", colors="darkorange"); a.grid(alpha=0.3)
    a5 = a.twinx()
    a5.plot(t_ns, hist[:, 6], "purple", lw=1.3, label="max E [eV]")
    a5.plot(t_ns, hist[:, 7] / 1e3, "gray", lw=1.3, ls="--", label="N_e makro [tys.]")
    a5.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=115)
    print(f"   zapisano: {out}")


def main():
    """Generuje wybrane rysunki podsumowujące, dla wersji jednowymiarowej lub na płaszczyźnie."""
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["1d", "2d"], default=None)
    args = p.parse_args()

    print("Generowanie figur wynikowych (przekroje czynne skalibrowane do LANDMARK)")
    if args.only != "2d":
        print(" [1D3V] ...")
        fig_1d(*run_1d())
    if args.only != "1d":
        print(" [2D3V z-r] ...")
        cfg, sim, sn, h, ne = run_2d("z-r")
        fig_2d(cfg, sim, sn, h, ne, "fig_2d3v_zr.png")
        print(" [2D3V z-theta] ...")
        cfg, sim, sn, h, ne = run_2d("z-theta")
        fig_2d(cfg, sim, sn, h, ne, "fig_2d3v_ztheta.png")
    print("Gotowe.")


if __name__ == "__main__":
    main()
