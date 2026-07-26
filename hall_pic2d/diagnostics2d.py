"""Okno podglądu dla wersji na płaszczyźnie: mapy zamiast profili.

W wersji jednowymiarowej rysowaliśmy krzywe wzdłuż osi. Tutaj, ponieważ
liczymy na płaszczyźnie, gęstość i potencjał pokazujemy jako mapy. Na mapie
gęstości widać ułożenie plazmy, na przykład strukturę wokół kanału albo
warstwy przy ściankach. Poza mapami mamy jeszcze rozkład elektronów w
przestrzeni położeń i prędkości, przebieg obwodu, rozkład energii oraz krótkie
podsumowanie liczbowe.
"""

import os
import numpy as np

from hall_pic.constants import E_CHARGE
from . import pusher2d


class LiveView2D:
    """Rysuje sześć wykresów podglądu wersji na płaszczyźnie albo zapisuje je jako obrazki."""

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
        self.fig.suptitle(
            f"PIC 2D3V - silnik Halla, geometria {cfg.geometry}, wiązki RE")
        self.axes[1, 2].axis("off")
        self.ax_V = self.axes[1, 0].twinx()      # zakładana raz, tak jak w wersji jednowymiarowej
        self._cb = [None, None]
        self.hist_t, self.hist_Id, self.hist_IL = [], [], []
        self.hist_VC, self.hist_Ire = [], []

    def push_history(self, t, Id, IL, VC, Ire):
        """Zapamiętuje kolejny punkt przebiegu obwodu do wykresu w czasie."""
        self.hist_t.append(t); self.hist_Id.append(Id); self.hist_IL.append(IL)
        self.hist_VC.append(VC); self.hist_Ire.append(Ire)

    def _extent(self):
        """Zwraca zakres osi mapy w milimetrach, potrzebny do jej narysowania."""
        cfg = self.cfg
        return [0.0, cfg.L2 * 1e3, 0.0, cfg.L1 * 1e3]

    def update(self, sim):
        """Przerysowuje wszystkie wykresy na podstawie bieżącego stanu symulacji."""
        if not self.enabled:
            return
        cfg, plt, ax = self.cfg, self.plt, self.axes
        el, ions = sim.electrons, sim.ions
        mm = 1e3

        # Pierwszy wykres: mapa gęstości elektronów.
        a = ax[0, 0]; a.clear()
        ne = pusher2d.deposit_number_density(el, cfg)
        im = a.imshow(ne, origin="lower", aspect="auto", extent=self._extent(),
                      cmap="viridis")
        a.set_title(f"n_e(x1,x2)   t = {sim.t*1e9:.1f} ns")
        a.set_xlabel(cfg.x2_label); a.set_ylabel("z [mm]")
        if self._cb[0] is None:
            self._cb[0] = self.fig.colorbar(im, ax=a, fraction=0.046)
        else:
            self._cb[0].update_normal(im)

        # Drugi wykres: mapa potencjału z liniami stałej wartości.
        a = ax[0, 1]; a.clear()
        im2 = a.imshow(sim.phi, origin="lower", aspect="auto",
                       extent=self._extent(), cmap="plasma")
        try:
            a.contour(np.linspace(0, cfg.L2*mm, cfg.n2_nodes),
                      np.linspace(0, cfg.L1*mm, cfg.n1_nodes),
                      sim.phi, levels=8, colors="w", linewidths=0.4, alpha=0.6)
        except Exception:
            pass
        a.set_title("φ(x1,x2) [V]")
        a.set_xlabel(cfg.x2_label); a.set_ylabel("z [mm]")
        if self._cb[1] is None:
            self._cb[1] = self.fig.colorbar(im2, ax=a, fraction=0.046)
        else:
            self._cb[1].update_normal(im2)

        # Trzeci wykres: elektrony w przestrzeni położeń i prędkości wzdłuż osi.
        a = ax[0, 2]; a.clear(); a.grid(alpha=0.3)
        a.set_title("Przestrzeń fazowa e⁻ (z, v_z)")
        a.set_xlabel("z [mm]"); a.set_ylabel("v_z [10⁶ m/s]")
        if el.N > 0:
            st = max(1, el.N // cfg.max_scatter_points)
            eps = 0.5*el.mass*(el.av1[::st]**2 + el.av2[::st]**2 + el.av3[::st]**2)/E_CHARGE
            a.scatter(el.ax1[::st]*mm, el.av1[::st]/1e6, c=eps, s=3, cmap="inferno",
                      vmin=0, vmax=max(cfg.E_RE_eV*4, 150), alpha=0.6)
            a.axhline(0, color="gray", lw=0.5)

        # Czwarty wykres: przebieg prądów i napięcia w obwodzie.
        a = ax[1, 0]; a.clear(); a.grid(alpha=0.3)
        a.set_title("Obwód RLC"); a.set_xlabel("t [ns]"); a.set_ylabel("I [A]")
        t_ns = np.array(self.hist_t)*1e9
        a.plot(t_ns, self.hist_Id, "b-", label="I_d")
        a.plot(t_ns, self.hist_IL, "r-", label="I_L")
        a.plot(t_ns, self.hist_Ire, color="orange", label="I_RE")
        a.legend(loc="upper left", fontsize=8)
        a4 = self.ax_V; a4.clear()
        a4.plot(t_ns, self.hist_VC, "k--", lw=1, alpha=0.6)
        a4.set_ylabel("V_C [V]")

        # Piąty wykres: rozkład energii elektronów z progiem wiązki.
        a = ax[1, 1]; a.clear(); a.grid(alpha=0.3)
        a.set_title("EEDF"); a.set_xlabel("energia [eV]"); a.set_ylabel("liczność (log)")
        if el.N > 0:
            a.hist(el.kinetic_energy_eV(), bins=80, weights=el.aw, log=True,
                   color="steelblue", alpha=0.85)
            a.axvline(cfg.E_RE_eV, color="red", ls="--",
                      label=f"E_RE={cfg.E_RE_eV:.0f} eV")
            a.legend(fontsize=8)

        # Szósty wykres: samo podsumowanie liczbowe w formie tekstu.
        a = ax[1, 2]; a.clear(); a.axis("off")
        a.text(0.02, 0.98, sim.stats_text(), va="top", ha="left",
               family="monospace", fontsize=9, transform=a.transAxes)

        if cfg.headless_save_dir:
            self.fig.savefig(
                os.path.join(cfg.headless_save_dir, f"frame_{sim.step:06d}.png"), dpi=80)
        else:
            self.fig.canvas.draw(); self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    def finalize(self):
        """Na końcu przebiegu zostawia otwarte okno, o ile nie pracujemy bez ekranu."""
        if not self.enabled or self.cfg.headless_save_dir:
            return
        self.plt.ioff(); self.plt.show()
