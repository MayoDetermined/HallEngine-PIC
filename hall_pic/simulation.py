"""Klasa Simulation — pętla główna PIC 1D3V silnika Halla.

Kolejność kroku (leapfrog / Boris):
  1. Depozycja ładunku rho (CIC).
  2. Poisson -> phi, E (BC: phi(0)=V_C z obwodu).
  3. Zbieranie E do cząstek, pchacz Borisa (E_x, B_y) -> nowe v, x.
  4. Warunki brzegowe: absorpcja anoda/katoda, akumulacja prądu wyładowania.
  5. Emisja elektronów z katody (neutralizator).
  6. Kolizje null-MCC (elektrony: elast./wzbudz./jonizacja; jony: CEX/elast.).
  7. APR (co apr_interval): rozrzedzanie tła + zagęszczanie wiązki RE.
  8. Aktualizacja obwodu RLC (RK4 z podkrokami) -> nowe V_C.
  9. Diagnostyka na żywo (co plot_interval).

Ważenie superczątstek: waga w [1/m^2] = n0*dx/n_ppc na starcie (spójna dla
elektronów i jonów -> quasineutralność). Prąd wyładowania:
  I_d = (Q_anoda_e + Q_anoda_i) * A_channel / dt   [A].
"""

import numpy as np

from .constants import E_CHARGE, M_ELECTRON
from .config import Config
from .species import Species
from .poisson import PoissonSolver
from .circuit import Circuit
from .collisions import NullCollisionMCC
from . import pusher, apr as apr_mod


class Simulation:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0.0
        self.step = 0

        # gatunki
        self.electrons = Species("e", -E_CHARGE, M_ELECTRON, capacity=400000)
        self.ions = Species("Xe+", +E_CHARGE, cfg.m_ion, capacity=400000)

        # waga referencyjna superczątstki [1/m^2]
        self.w_ref = cfg.n0_plasma * cfg.dx / cfg.n_ppc

        self.poisson = PoissonSolver(cfg)
        self.circuit = Circuit(cfg)
        self.mcc = NullCollisionMCC(cfg, self.rng)

        self.phi = np.zeros(cfg.n_nodes)
        self.E = np.zeros(cfg.n_nodes)

        # akumulatory prądu (na krok)
        self.I_d = 0.0
        self.I_RE = 0.0
        self.n_split = 0
        self.n_merge = 0
        self.n_ionized = 0
        # liczniki skumulowane (od startu symulacji)
        self.total_split = 0
        self.total_merge = 0
        self.total_ionized = 0

        self._seed_plasma()

    # ---------------- inicjalizacja plazmy ----------------
    def _seed_plasma(self):
        cfg = self.cfg
        n_cell = cfg.n_ppc
        N = cfg.Nx * n_cell
        # równomierne położenia w każdej komórce
        cell = np.repeat(np.arange(cfg.Nx), n_cell)
        xr = self.rng.random(N)
        x = (cell + xr) * cfg.dx

        vth_e = np.sqrt(E_CHARGE * cfg.Te0_eV / M_ELECTRON)
        vth_i = np.sqrt(E_CHARGE * cfg.Ti0_eV / cfg.m_ion)

        # elektrony
        vex = self.rng.normal(0, vth_e, N)
        vey = self.rng.normal(0, vth_e, N)
        vez = self.rng.normal(0, vth_e, N)
        self.electrons.add(x, vex, vey, vez, np.full(N, self.w_ref))
        # jony (te same położenia -> quasineutralność startowa)
        vix = self.rng.normal(0, vth_i, N)
        viy = self.rng.normal(0, vth_i, N)
        viz = self.rng.normal(0, vth_i, N)
        self.ions.add(x.copy(), vix, viy, viz, np.full(N, self.w_ref))

    # ---------------- pojedynczy krok ----------------
    def step_once(self):
        cfg = self.cfg
        # 1. depozycja
        rho = pusher.deposit_charge([self.electrons, self.ions], cfg)
        # 2. Poisson (BC z obwodu)
        self.phi, self.E = self.poisson.solve(rho, self.circuit.V_C)
        # 3. push
        Ee = pusher.gather_field(self.electrons, self.E, cfg)
        Ei = pusher.gather_field(self.ions, self.E, cfg)
        pusher.boris_push(self.electrons, Ee, cfg)
        pusher.boris_push(self.ions, Ei, cfg)
        # 4. brzegi + prąd wyładowania
        qa_e, qc_e, _ = pusher.apply_boundaries(self.electrons, cfg)
        qa_i, qc_i, n_i_cath = pusher.apply_boundaries(self.ions, cfg)
        # prąd przewodzenia na anodzie [A]; ładunek elektronu ujemny -> znak
        Q_anode = qa_e + qa_i             # [C/m^2]
        self.I_d = -Q_anode * cfg.A_channel / cfg.dt
        # 5. emisja katody: kompensuje jony docierające do katody
        inj_w = cfg.cathode_gain * n_i_cath
        pusher.inject_cathode_electrons(self.electrons, self.w_ref, inj_w, cfg, self.rng)
        # 6. kolizje
        if cfg.enable_collisions:
            self.n_ionized = self.mcc.collide_electrons(self.electrons, self.ions, self.w_ref)
            self.mcc.collide_ions(self.ions, self.w_ref)
            self.total_ionized += self.n_ionized
        # 7. APR
        if cfg.enable_apr and (self.step % cfg.apr_interval == 0):
            self.n_split, self.n_merge = apr_mod.run_apr(self.electrons, cfg, self.w_ref, self.rng)
            self.total_split += self.n_split
            self.total_merge += self.n_merge
        # 8. obwód
        self.circuit.advance(self.I_d, cfg.dt)
        # prąd wiązki RE (elektrony > E_RE przekraczające płaszczyznę wylotu — miara)
        self.I_RE = self._beam_current()

        self.t += cfg.dt
        self.step += 1

    def _beam_current(self):
        """Szacunkowy prąd niesiony przez elektrony RE (energia > E_RE) [A]."""
        el = self.electrons
        if el.N == 0:
            return 0.0
        eps = el.kinetic_energy_eV()
        mask = eps > self.cfg.E_RE_eV
        if not np.any(mask):
            return 0.0
        # prąd = q * A * suma(w * v_x)/L  (średni transport ładunku RE)
        flux = np.sum(el.aw[mask] * el.avx[mask]) / self.cfg.L
        return abs(E_CHARGE * self.cfg.A_channel * flux)

    # ---------------- diagnostyka pomocnicza ----------------
    def stats_text(self):
        el = self.electrons
        eps = el.kinetic_energy_eV() if el.N > 0 else np.array([0.0])
        n_re = int(np.sum(eps > self.cfg.E_RE_eV)) if el.N > 0 else 0
        mean_eps = float(np.average(eps, weights=el.aw)) if el.N > 0 else 0.0
        max_eps = float(np.max(eps)) if el.N > 0 else 0.0
        return (
            f"krok         : {self.step}/{self.cfg.n_steps}\n"
            f"t            : {self.t*1e9:8.2f} ns\n"
            f"N_e (makro)  : {el.N}\n"
            f"N_i (makro)  : {self.ions.N}\n"
            f"---- obwód ----\n"
            f"I_d          : {self.I_d:8.3f} A\n"
            f"I_L          : {self.circuit.I_L:8.3f} A\n"
            f"V_C (anoda)  : {self.circuit.V_C:8.2f} V\n"
            f"---- wiązka RE ----\n"
            f"E_RE próg    : {self.cfg.E_RE_eV:6.1f} eV\n"
            f"N_e(RE)      : {n_re}\n"
            f"I_RE         : {self.I_RE:8.4f} A\n"
            f"<E_e>        : {mean_eps:8.2f} eV\n"
            f"max E_e      : {max_eps:8.1f} eV\n"
            f"---- APR/MCC (skumul.) ----\n"
            f"split/merge  : {self.total_split} / {self.total_merge}\n"
            f"jonizacje    : {self.total_ionized}\n"
            f"nu_max_e·dt  : {self.mcc.nu_max_e*self.cfg.dt:.3e}\n"
        )
