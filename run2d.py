"""Uruchamianie symulacji w wersji na płaszczyźnie.

Przykłady wywołania:
    python run2d.py                              # płaszczyzna wokół kanału, podgląd na żywo
    python run2d.py --geometry z-r               # płaszczyzna w poprzek kanału, ze ściankami
    python run2d.py --geometry z-r --steps 2000  # krótki przebieg testowy
    python run2d.py --headless out2d/            # bez okna, z zapisem klatek
    python run2d.py --n1 96 --n2 48 --ppc 12     # rzadsza siatka, dla szybszego przebiegu
"""

import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from hall_pic2d import Config2D, Simulation2D
from hall_pic2d.diagnostics2d import LiveView2D


def parse_args():
    """Czyta ustawienia podane w wierszu poleceń."""
    p = argparse.ArgumentParser(description="PIC 2D3V silnika Halla")
    p.add_argument("--geometry", choices=["z-theta", "z-r"], default="z-theta",
                   help="płaszczyzna symulacji")
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n1", type=int, default=None, help="komórki wzdłuż z")
    p.add_argument("--n2", type=int, default=None, help="komórki wzdłuż x2")
    p.add_argument("--ppc", type=int, default=None, help="cząstek na komórkę")
    p.add_argument("--headless", type=str, default="")
    p.add_argument("--no-apr", action="store_true")
    p.add_argument("--no-collisions", action="store_true")
    p.add_argument("--plot-interval", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    """Składa konfigurację z podanych ustawień, uruchamia przebieg i podgląd."""
    args = parse_args()
    cfg = Config2D()
    cfg.geometry = args.geometry
    if args.t_end is not None: cfg.t_end = args.t_end
    if args.dt is not None: cfg.dt = args.dt
    if args.n1 is not None: cfg.N1 = args.n1
    if args.n2 is not None: cfg.N2 = args.n2
    if args.ppc is not None: cfg.n_ppc = args.ppc
    if args.plot_interval is not None: cfg.plot_interval = args.plot_interval
    if args.seed is not None: cfg.seed = args.seed
    if args.no_apr: cfg.enable_apr = False
    if args.no_collisions: cfg.enable_collisions = False
    if args.headless:
        cfg.headless_save_dir = args.headless
        cfg.live_view = False
    cfg.__post_init__()

    n_steps = args.steps if args.steps is not None else cfg.n_steps

    print("=" * 66)
    print(f" PIC 2D3V - silnik Halla, geometria {cfg.geometry}")
    print("=" * 66)
    print(f" x1 = z      : {cfg.L1*1e3:.1f} mm, N1 = {cfg.N1}, h1 = {cfg.h1*1e6:.1f} um")
    print(f" x2          : {cfg.L2*1e3:.1f} mm, N2 = {cfg.N2}, h2 = {cfg.h2*1e6:.1f} um"
          f"  [{cfg.x2_bc}]")
    print(f" B_max       : {cfg.B_max*1e4:.0f} G @ z = {cfg.x_B*1e3:.1f} mm")
    print(f" dt          : {cfg.dt*1e12:.2f} ps, kroków = {n_steps}")
    print(f" cząstki     : {cfg.n_cells*cfg.n_ppc} / gatunek (ppc = {cfg.n_ppc})")
    print(f" w_ref       : {cfg.n0_plasma*cfg.h1*cfg.h2/cfg.n_ppc:.3e} 1/m")
    print(f" current_fac : {cfg.current_factor:.4f} m")
    print(f" APR: {'ON' if cfg.enable_apr else 'OFF'}   "
          f"MCC: {'ON' if cfg.enable_collisions else 'OFF'}")
    print("=" * 66)

    sim = Simulation2D(cfg)
    view = LiveView2D(cfg)

    t0 = time.time()
    for _ in range(n_steps):
        sim.step_once()
        if sim.step % max(1, cfg.plot_interval // 5) == 0:
            view.push_history(sim.t, sim.I_d, sim.circuit.I_L, sim.circuit.V_C, sim.I_RE)
        if sim.step % cfg.plot_interval == 0:
            view.update(sim)
            wall = time.time() - t0
            print(f" krok {sim.step:6d}  t={sim.t*1e9:7.2f} ns  N_e={sim.electrons.N:7d}"
                  f"  I_d={sim.I_d:7.2f} A  V_C={sim.circuit.V_C:6.1f} V"
                  f"  I_RE={sim.I_RE:7.4f} A  [{sim.step/max(wall,1e-9):.1f} kr/s]")

    print("=" * 66)
    print(f" Zakończono: {sim.step} kroków, t = {sim.t*1e9:.1f} ns, "
          f"czas ściany = {time.time()-t0:.1f} s")
    view.update(sim)
    view.finalize()


if __name__ == "__main__":
    main()
