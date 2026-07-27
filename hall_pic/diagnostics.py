"""Okno podglądu, w którym na żywo widać, jak rośnie wiązka elektronów.

Okno składa się z sześciu wykresów odświeżanych co pewną liczbę kroków. Widać
na nich kolejno: rozkład elektronów w przestrzeni położeń i prędkości, gdzie
wiązka odrywa się od tła jako osobna smuga; gęstości elektronów i jonów wraz z
kształtem pola magnetycznego; potencjał i pole elektryczne; przebieg prądów i
napięcia w obwodzie; rozkład energii elektronów z zaznaczonym progiem wiązki;
oraz krótkie podsumowanie liczbowe. Jeśli zamiast okna podamy katalog, obrazki
będą zapisywane do plików, co przydaje się przy pracy bez ekranu.
"""

import os
import numpy as np

from .constants import E_CHARGE
from . import pusher


class LiveView:
    """Rysuje sześć wykresów podglądu albo zapisuje je jako kolejne obrazki."""

    def __init__(self, cfg):
        """Zakłada okno z wykresami albo przygotowuje zapis klatek do plików."""
        self.cfg = cfg
        self.enabled = cfg.live_view or bool(cfg.headless_save_dir)
        if not self.enabled:
            return
        import matplotlib
        if cfg.headless_save_dir:
            matplotlib.use("Agg")
            os.makedirs(cfg.headless_save_dir, exist_ok=True)
        import matplotlib.pyplot as plt
        self.plt = plt
        if cfg.live_view and not cfg.headless_save_dir:
            plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 8))
        self.fig.suptitle("PIC 1D3V - silnik Halla (SPT-100), formowanie wiązek RE")
        self._init_artists()
        self.hist_t = []
        self.hist_Id = []
        self.hist_IL = []
        self.hist_VC = []
        self.hist_Ire = []

    def _init_artists(self):
        """Nadaje wykresom tytuły i opisy osi oraz zakłada osie pomocnicze."""
        ax = self.axes
        for a in ax.ravel():
            a.grid(alpha=0.3)
        ax[0, 0].set_title("Przestrzeń fazowa e⁻ (x, v_x)")
        ax[0, 0].set_xlabel("x [mm]"); ax[0, 0].set_ylabel("v_x [10⁶ m/s]")
        ax[0, 1].set_title("Gęstości i pole B")
        ax[0, 1].set_xlabel("x [mm]"); ax[0, 1].set_ylabel("n [m⁻³]")
        ax[0, 2].set_title("Potencjał i pole E")
        ax[0, 2].set_xlabel("x [mm]"); ax[0, 2].set_ylabel("φ [V]")
        ax[1, 0].set_title("Obwód RLC")
        ax[1, 0].set_xlabel("t [ns]"); ax[1, 0].set_ylabel("I [A]")
        ax[1, 1].set_title("EEDF (rozkład energii e⁻)")
        ax[1, 1].set_xlabel("energia [eV]"); ax[1, 1].set_ylabel("liczność (log)")
        ax[1, 2].set_title("Statystyki")
        ax[1, 2].axis("off")
        # Osie pomocnicze zakładamy tylko raz i czyścimy przy każdej klatce.
        # Gdybyśmy tworzyli je za każdym razem od nowa, nakładałyby się na
        # siebie, mnożąc opisy i linie.
        self.ax_B = ax[0, 1].twinx()     # dodatkowa oś na pole magnetyczne
        self.ax_E = ax[0, 2].twinx()     # dodatkowa oś na pole elektryczne
        self.ax_V = ax[1, 0].twinx()     # dodatkowa oś na napięcie anody

    def update(self, sim):
        """Przerysowuje wszystkie wykresy na podstawie bieżącego stanu symulacji."""
        if not self.enabled:
            return
        cfg = self.cfg
        plt = self.plt
        ax = self.axes
        el = sim.electrons
        ions = sim.ions
        mm = 1e3

        # Pierwszy wykres: elektrony w przestrzeni położeń i prędkości.
        a = ax[0, 0]; a.clear(); a.grid(alpha=0.3)
        a.set_title(f"Przestrzeń fazowa e⁻   t = {sim.t*1e9:.1f} ns")
        a.set_xlabel("x [mm]"); a.set_ylabel("v_x [10⁶ m/s]")
        if el.N > 0:
            n = el.N
            step = max(1, n // cfg.max_scatter_points)
            xs_ = el.ax[::step]
            vxs = el.avx[::step]
            eps = 0.5 * el.mass * (el.avx[::step]**2 + el.avy[::step]**2 + el.avz[::step]**2) / E_CHARGE
            sc = a.scatter(xs_*mm, vxs/1e6, c=eps, s=3, cmap="inferno",
                           vmin=0, vmax=max(cfg.E_RE_eV*4, 150), alpha=0.6)
            a.axhline(0, color="gray", lw=0.5)

        # Drugi wykres: gęstości obu gatunków i kształt pola magnetycznego.
        a = ax[0, 1]; a.clear(); a.grid(alpha=0.3)
        a.set_title("Gęstości i pole B"); a.set_xlabel("x [mm]"); a.set_ylabel("n [m⁻³]")
        ne = pusher.deposit_number_density(el, cfg)
        ni = pusher.deposit_number_density(ions, cfg)
        a.plot(cfg.x_nodes*mm, ne, "b-", label="n_e", lw=1.2)
        a.plot(cfg.x_nodes*mm, ni, "r-", label="n_i", lw=1.2)
        a.legend(loc="upper right", fontsize=8)
        a2 = self.ax_B; a2.clear()
        a2.plot(cfg.x_nodes*mm, cfg.B_profile(cfg.x_nodes)*1e4, "g--", lw=1, alpha=0.6)
        a2.set_ylabel("B [G]", color="g")

        # Trzeci wykres: potencjał i pole elektryczne.
        a = ax[0, 2]; a.clear(); a.grid(alpha=0.3)
        a.set_title("Potencjał i pole E"); a.set_xlabel("x [mm]"); a.set_ylabel("φ [V]")
        a.plot(cfg.x_nodes*mm, sim.phi, "m-", lw=1.5, label="φ")
        a.legend(loc="upper right", fontsize=8)
        a3 = self.ax_E; a3.clear()
        a3.plot(cfg.x_nodes*mm, sim.E/1e3, "c-", lw=1, alpha=0.6)
        a3.set_ylabel("E_x [kV/m]", color="c")

        # Czwarty wykres: przebieg prądów i napięcia w obwodzie.
        a = ax[1, 0]; a.clear(); a.grid(alpha=0.3)
        a.set_title("Obwód RLC"); a.set_xlabel("t [ns]"); a.set_ylabel("I [A]")
        t_ns = np.array(self.hist_t)*1e9
        a.plot(t_ns, self.hist_Id, "b-", label="I_d (wyładowanie)")
        a.plot(t_ns, self.hist_IL, "r-", label="I_L (obwód)")
        a.plot(t_ns, self.hist_Ire, "orange", label="I_RE (wiązka)")
        a.legend(loc="upper left", fontsize=8)
        a4 = self.ax_V; a4.clear()
        a4.plot(t_ns, self.hist_VC, "k--", lw=1, alpha=0.6)
        a4.set_ylabel("V_C [V]")

        # Piąty wykres: rozkład energii elektronów z progiem wiązki.
        a = ax[1, 1]; a.clear(); a.grid(alpha=0.3)
        a.set_title("EEDF"); a.set_xlabel("energia [eV]"); a.set_ylabel("liczność (log)")
        if el.N > 0:
            eps_all = el.kinetic_energy_eV()
            # Zakres osi ustawiamy jawnie, od zera do najwyższej energii,
            # tak by prawy kraniec histogramu pokrywał się z odczytem najwyższej
            # energii, a próg wiązki zawsze mieścił się w widoku.
            e_top = max(float(eps_all.max()), cfg.E_RE_eV * 1.5)
            a.hist(eps_all, bins=80, range=(0.0, e_top), weights=el.aw, log=True,
                   color="steelblue", alpha=0.8)
            a.set_xlim(0.0, e_top)
            a.axvline(cfg.E_RE_eV, color="red", ls="--", label=f"E_RE={cfg.E_RE_eV:.0f} eV")
            a.legend(fontsize=8)

        # Szósty wykres: samo podsumowanie liczbowe w formie tekstu.
        a = ax[1, 2]; a.clear(); a.axis("off")
        txt = sim.stats_text()
        a.text(0.02, 0.98, txt, va="top", ha="left", family="monospace",
               fontsize=8, transform=a.transAxes)

        if cfg.headless_save_dir:
            fn = os.path.join(cfg.headless_save_dir, f"frame_{sim.step:06d}.png")
            self.fig.savefig(fn, dpi=80)
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def push_history(self, t, Id, IL, VC, Ire):
        """Zapamiętuje kolejny punkt przebiegu obwodu do wykresu w czasie."""
        self.hist_t.append(t)
        self.hist_Id.append(Id)
        self.hist_IL.append(IL)
        self.hist_VC.append(VC)
        self.hist_Ire.append(Ire)

    def finalize(self):
        """Na końcu przebiegu zostawia otwarte okno, o ile nie pracujemy bez ekranu."""
        if not self.enabled or self.cfg.headless_save_dir:
            return
        self.plt.ioff()
        self.plt.show()
