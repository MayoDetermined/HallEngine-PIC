"""Pętla główna PIC 2D3V silnika Halla (geometria przełączalna z-θ / z-r).

Kolejność kroku identyczna jak w 1D — zmienia się wymiarowość operacji siatkowych:
  1. depozycja ρ (CIC dwuliniowa)
  2. Poisson 2D -> φ, E1, E2   (BC osiowe: φ(0)=V_C z obwodu)
  3. zbieranie (E1,E2) + pchacz Borisa z pełnym 3-wektorem B
  4. brzegi (anoda/katoda; ścianki albo periodyczność) + prąd wyładowania
  5. emisja z katody
  6. null-MCC (neutrale zamrożone)
  7. APR (co apr_interval)
  8. obwód RLC
"""

import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON
from hall_pic.circuit import Circuit
from .config2d import Config2D
from .species2d import Species2D
from .poisson2d import Poisson2D
from .collisions2d import NullCollisionMCC2D
from . import pusher2d, apr2d


class Simulation2D:
    def __init__(self, cfg: Config2D):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0.0
        self.step = 0

        self.electrons = Species2D("e", -E_CHARGE, M_ELECTRON, capacity=400000)
        self.ions = Species2D("Xe+", +E_CHARGE, cfg.m_ion, capacity=400000)

        # waga referencyjna [1/m]: n0 = w·n_ppc/(h1·h2)
        self.w_ref = cfg.n0_plasma * cfg.h1 * cfg.h2 / cfg.n_ppc

        self.poisson = Poisson2D(cfg)
        self.circuit = Circuit(cfg)
        self.mcc = NullCollisionMCC2D(cfg, self.rng)

        shape = (cfg.n1_nodes, cfg.n2_nodes)
        self.phi = np.zeros(shape)
        self.E1 = np.zeros(shape)
        self.E2 = np.zeros(shape)

        self.I_d = 0.0
        self.I_RE = 0.0
        self.n_split = self.n_merge = self.n_ionized = 0
        self.total_split = self.total_merge = self.total_ionized = 0
        self.wall_loss_w = 0.0          # skumulowana waga utracona na ściankach (z-r)

        self._seed_plasma()

    # ---------------- inicjalizacja ----------------
    def _seed_plasma(self):
        cfg = self.cfg
        N = cfg.n_cells * cfg.n_ppc
        rng = self.rng
        cell = np.repeat(np.arange(cfg.n_cells), cfg.n_ppc)
        i = cell // cfg.N2
        j = cell % cfg.N2
        x1 = (i + rng.random(N)) * cfg.h1
        x2 = (j + rng.random(N)) * cfg.h2

        vth_e = np.sqrt(E_CHARGE * cfg.Te0_eV / M_ELECTRON)
        vth_i = np.sqrt(E_CHARGE * cfg.Ti0_eV / cfg.m_ion)
        w = np.full(N, self.w_ref)

        self.electrons.add(x1, x2,
                           rng.normal(0, vth_e, N), rng.normal(0, vth_e, N),
                           rng.normal(0, vth_e, N), w)
        self.ions.add(x1.copy(), x2.copy(),
                      rng.normal(0, vth_i, N), rng.normal(0, vth_i, N),
                      rng.normal(0, vth_i, N), w.copy())

    # ---------------- krok ----------------
    def step_once(self):
        cfg = self.cfg
        # wagi CIC liczone RAZ na gatunek i reużywane przez depozycję i zbieranie
        # (obie operacje działają na tych samych położeniach, przed pchaczem)
        cache = [pusher2d.cic_weights(self.electrons, cfg),
                 pusher2d.cic_weights(self.ions, cfg)]
        rho = pusher2d.deposit_charge([self.electrons, self.ions], cfg, cached=cache)
        self.phi, self.E1, self.E2 = self.poisson.solve(rho, self.circuit.V_C)

        e1, e2 = pusher2d.gather_field(self.electrons, self.E1, self.E2, cfg, cached=cache[0])
        pusher2d.boris_push(self.electrons, e1, e2, cfg)
        i1, i2 = pusher2d.gather_field(self.ions, self.E1, self.E2, cfg, cached=cache[1])
        pusher2d.boris_push(self.ions, i1, i2, cfg)

        qa_e, _, _, wall_e = pusher2d.apply_boundaries(self.electrons, cfg)
        qa_i, _, w_cath_i, wall_i = pusher2d.apply_boundaries(self.ions, cfg)
        self.wall_loss_w += wall_e + wall_i

        # prąd wyładowania: waga [1/m] × current_factor [m] -> cząstki rzeczywiste
        self.I_d = -(qa_e + qa_i) * cfg.current_factor / cfg.dt

        pusher2d.inject_cathode_electrons(
            self.electrons, self.w_ref, cfg.cathode_gain * w_cath_i, cfg, self.rng)

        if cfg.enable_collisions:
            self.n_ionized = self.mcc.collide_electrons(self.electrons, self.ions)
            self.mcc.collide_ions(self.ions)
            self.total_ionized += self.n_ionized

        if cfg.enable_apr and (self.step % cfg.apr_interval == 0):
            self.n_split, self.n_merge = apr2d.run_apr(
                self.electrons, cfg, self.w_ref, self.rng)
            self.total_split += self.n_split
            self.total_merge += self.n_merge

        self.circuit.advance(self.I_d, cfg.dt)
        self.I_RE = self._beam_current()

        self.t += cfg.dt
        self.step += 1

    def _beam_current(self):
        """Prąd niesiony przez elektrony RE (E > E_RE), transport osiowy."""
        el = self.electrons
        if el.N == 0:
            return 0.0
        eps = el.kinetic_energy_eV()
        m = eps > self.cfg.E_RE_eV
        if not np.any(m):
            return 0.0
        flux = np.sum(el.aw[m] * el.av1[m]) / self.cfg.L1
        return abs(E_CHARGE * self.cfg.current_factor * flux)

    # ---------------- diagnostyka ----------------
    def stats_text(self):
        cfg = self.cfg
        el = self.electrons
        eps = el.kinetic_energy_eV() if el.N > 0 else np.array([0.0])
        n_re = int(np.sum(eps > cfg.E_RE_eV)) if el.N > 0 else 0
        mean_eps = float(np.average(eps, weights=el.aw)) if el.N > 0 else 0.0
        max_eps = float(np.max(eps)) if el.N > 0 else 0.0
        wall = "n/d (periodyczny)" if cfg.x2_bc == "periodic" else f"{self.wall_loss_w:.3e}"
        return (
            f"geometria    : {cfg.geometry}\n"
            f"siatka       : {cfg.N1} x {cfg.N2}\n"
            f"krok         : {self.step}/{cfg.n_steps}\n"
            f"t            : {self.t*1e9:8.2f} ns\n"
            f"N_e (makro)  : {el.N}\n"
            f"N_i (makro)  : {self.ions.N}\n"
            f"---- obwód ----\n"
            f"I_d          : {self.I_d:8.3f} A\n"
            f"I_L          : {self.circuit.I_L:8.3f} A\n"
            f"V_C (anoda)  : {self.circuit.V_C:8.2f} V\n"
            f"---- wiązka RE ----\n"
            f"E_RE próg    : {cfg.E_RE_eV:6.1f} eV\n"
            f"N_e(RE)      : {n_re}\n"
            f"I_RE         : {self.I_RE:8.4f} A\n"
            f"<E_e>        : {mean_eps:8.2f} eV\n"
            f"max E_e      : {max_eps:8.1f} eV\n"
            f"---- APR/MCC (skumul.) ----\n"
            f"split/merge  : {self.total_split} / {self.total_merge}\n"
            f"jonizacje    : {self.total_ionized}\n"
            f"strata ścian : {wall}\n"
        )
