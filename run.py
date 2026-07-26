"""Uruchamianie symulacji w wersji jednowymiarowej.

Przykłady wywołania:
    python run.py                      # podgląd na żywo, przebieg do dwustu nanosekund
    python run.py --t-end 1e-6         # dłuższy przebieg, aż do jednej mikrosekundy
    python run.py --headless out/      # bez okna, z zapisem klatek do katalogu
    python run.py --no-apr             # bez dzielenia i łączenia czastek
    python run.py --steps 2000         # krótki przebieg testowy o zadanej liczbie kroków
"""

import argparse
import sys
import time

# Konsola na Windows bywa ustawiona na kodowanie, które nie zna polskich znaków
# ani znaków jednostek. Prosimy więc o kodowanie, które je obsługuje.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from hall_pic import Config, Simulation
from hall_pic.diagnostics import LiveView


def parse_args():
    """Czyta ustawienia podane w wierszu poleceń."""
    p = argparse.ArgumentParser(description="PIC 1D3V silnika Halla (SPT-100)")
    p.add_argument("--t-end", type=float, default=None, help="czas końcowy [s]")
    p.add_argument("--dt", type=float, default=None, help="krok czasowy [s]")
    p.add_argument("--steps", type=int, default=None, help="limit liczby kroków")
    p.add_argument("--nx", type=int, default=None, help="liczba komórek siatki")
    p.add_argument("--ppc", type=int, default=None, help="cząstek na komórkę (start)")
    p.add_argument("--headless", type=str, default="", help="katalog na klatki PNG (tryb bez okna)")
    p.add_argument("--no-apr", action="store_true", help="wyłącz APR")
    p.add_argument("--no-collisions", action="store_true", help="wyłącz kolizje MCC")
    p.add_argument("--plot-interval", type=int, default=None, help="co ile kroków rysować")
    p.add_argument("--seed", type=int, default=None, help="ziarno RNG")
    return p.parse_args()


def main():
    """Składa konfigurację z podanych ustawień, uruchamia przebieg i podgląd."""
    args = parse_args()
    cfg = Config()
    if args.t_end is not None: cfg.t_end = args.t_end
    if args.dt is not None: cfg.dt = args.dt
    if args.nx is not None: cfg.Nx = args.nx
    if args.ppc is not None: cfg.n_ppc = args.ppc
    if args.plot_interval is not None: cfg.plot_interval = args.plot_interval
    if args.seed is not None: cfg.seed = args.seed
    if args.no_apr: cfg.enable_apr = False
    if args.no_collisions: cfg.enable_collisions = False
    if args.headless:
        cfg.headless_save_dir = args.headless
        cfg.live_view = False
    cfg.__post_init__()  # po nadpisaniu ustawień trzeba przeliczyć to, co z nich wynika

    n_steps = args.steps if args.steps is not None else cfg.n_steps

    print("=" * 60)
    print(" PIC 1D3V - silnik Halla (SPT-100)")
    print("=" * 60)
    print(f" Domena L         : {cfg.L*1e3:.1f} mm, Nx = {cfg.Nx}, dx = {cfg.dx*1e6:.2f} um")
    print(f" B_max            : {cfg.B_max*1e4:.0f} G @ x = {cfg.x_B*1e3:.1f} mm")
    print(f" dt               : {cfg.dt*1e12:.2f} ps, kroków = {n_steps}")
    print(f" waga w_ref       : {cfg.n0_plasma*cfg.dx/cfg.n_ppc:.3e} 1/m^2")
    print(f" obwód RLC        : V_ps={cfg.V_ps} V, R={cfg.R_circ} Ω, "
          f"L={cfg.L_circ*1e6:.0f} uH, C={cfg.C_circ*1e9:.0f} nF")
    print(f" APR              : {'ON' if cfg.enable_apr else 'OFF'}, "
          f"MCC: {'ON' if cfg.enable_collisions else 'OFF'}")
    print("=" * 60)

    sim = Simulation(cfg)
    view = LiveView(cfg)

    t0 = time.time()
    for i in range(n_steps):
        sim.step_once()
        # Punkty do wykresu obwodu zbieramy rzadziej, żeby lista nie puchła.
        if sim.step % max(1, cfg.plot_interval // 5) == 0:
            view.push_history(sim.t, sim.I_d, sim.circuit.I_L, sim.circuit.V_C, sim.I_RE)
        if sim.step % cfg.plot_interval == 0:
            view.update(sim)
            wall = time.time() - t0
            rate = sim.step / wall if wall > 0 else 0
            print(f" krok {sim.step:6d}  t={sim.t*1e9:7.2f} ns  "
                  f"N_e={sim.electrons.N:6d}  I_d={sim.I_d:6.2f} A  "
                  f"V_C={sim.circuit.V_C:6.1f} V  I_RE={sim.I_RE:7.4f} A  "
                  f"[{rate:.0f} kroków/s]")

    print("=" * 60)
    print(f" Zakończono: {sim.step} kroków, t = {sim.t*1e9:.1f} ns, "
          f"czas ściany = {time.time()-t0:.1f} s")
    view.update(sim)
    view.finalize()


if __name__ == "__main__":
    main()
