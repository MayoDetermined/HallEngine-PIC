"""Główna pętla symulacji, która spina ze sobą wszystkie części.

Jeden krok w czasie przebiega zawsze tak samo. Najpierw rozkładamy ładunek
czastek na siatkę i wyznaczamy z niego potencjał oraz pole, przy czym napięcie
anody bierzemy z obwodu. Potem odczytujemy pole w miejscu każdej czastki i
przesuwamy czastki. Następnie pochłaniamy te, które trafiły na elektrody, i z
zebranego ładunku odczytujemy prąd wyładowania. Katoda dosyła świeże elektrony,
rozgrywamy zderzenia z gazem, co pewien czas dzielimy i łączymy czastki, na
koniec przesuwamy obwód i odświeżamy podgląd.

Wagę czastek dobieramy na starcie tak samo dla elektronów i jonów, dzięki czemu
plazma zaczyna od stanu obojętnego elektrycznie. Prąd wyładowania liczymy z
ładunku, jaki w danym kroku trafił na anodę.
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
    """Prowadzi symulację krok po kroku, łącząc pola, czastki, zderzenia i obwód."""

    def __init__(self, cfg: Config):
        """Przygotowuje pola, oba gatunki czastek, solver, obwód i plazmę startową."""
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0.0
        self.step = 0

        # Dwa gatunki czastek: elektrony i jony ksenonu.
        self.electrons = Species("e", -E_CHARGE, M_ELECTRON, capacity=400000)
        self.ions = Species("Xe+", +E_CHARGE, cfg.m_ion, capacity=400000)

        # Wspólna waga wzorcowa czastki na starcie.
        self.w_ref = cfg.n0_plasma * cfg.dx / cfg.n_ppc

        self.poisson = PoissonSolver(cfg)
        self.circuit = Circuit(cfg)
        self.mcc = NullCollisionMCC(cfg, self.rng)

        self.phi = np.zeros(cfg.n_nodes)
        self.E = np.zeros(cfg.n_nodes)

        # Wielkości liczone na bieżąco w każdym kroku.
        self.I_d = 0.0
        self.I_RE = 0.0
        self.n_split = 0
        self.n_merge = 0
        self.n_ionized = 0
        # Liczniki zbierane od początku przebiegu.
        self.total_split = 0
        self.total_merge = 0
        self.total_ionized = 0

        self._seed_plasma()

    # Rozstawienie plazmy początkowej.
    def _seed_plasma(self):
        """Rozstawia plazmę startową: elektrony i jony w tych samych miejscach.

        Dzięki wspólnym położeniom plazma zaczyna od stanu obojętnego
        elektrycznie. Prędkości losujemy zgodnie z zadanymi temperaturami.
        """
        cfg = self.cfg
        n_cell = cfg.n_ppc
        N = cfg.Nx * n_cell
        # Czastki rozkładamy równomiernie wewnątrz każdej komórki.
        cell = np.repeat(np.arange(cfg.Nx), n_cell)
        xr = self.rng.random(N)
        x = (cell + xr) * cfg.dx

        vth_e = np.sqrt(E_CHARGE * cfg.Te0_eV / M_ELECTRON)
        vth_i = np.sqrt(E_CHARGE * cfg.Ti0_eV / cfg.m_ion)

        # Elektrony.
        vex = self.rng.normal(0, vth_e, N)
        vey = self.rng.normal(0, vth_e, N)
        vez = self.rng.normal(0, vth_e, N)
        self.electrons.add(x, vex, vey, vez, np.full(N, self.w_ref))
        # Jony w tych samych miejscach, żeby plazma startowała jako obojętna.
        vix = self.rng.normal(0, vth_i, N)
        viy = self.rng.normal(0, vth_i, N)
        viz = self.rng.normal(0, vth_i, N)
        self.ions.add(x.copy(), vix, viy, viz, np.full(N, self.w_ref))

    # Pojedynczy krok w czasie.
    def step_once(self):
        """Wykonuje jeden pełny krok symulacji w czasie."""
        cfg = self.cfg
        # Rozłożenie ładunku czastek na siatkę.
        rho = pusher.deposit_charge([self.electrons, self.ions], cfg)
        # Potencjał i pole, z napięciem anody wziętym z obwodu.
        self.phi, self.E = self.poisson.solve(rho, self.circuit.V_C)
        # Odczyt pola w miejscu czastek i ich przesunięcie.
        Ee = pusher.gather_field(self.electrons, self.E, cfg)
        Ei = pusher.gather_field(self.ions, self.E, cfg)
        pusher.boris_push(self.electrons, Ee, cfg)
        pusher.boris_push(self.ions, Ei, cfg)
        # Pochłonięcie czastek na elektrodach i odczyt prądu wyładowania.
        qa_e, qc_e, _ = pusher.apply_boundaries(self.electrons, cfg)
        qa_i, qc_i, n_i_cath = pusher.apply_boundaries(self.ions, cfg)
        # Ładunek zebrany na anodzie zamieniamy na prąd. Znak wynika z tego,
        # że ładunek elektronu jest ujemny.
        Q_anode = qa_e + qa_i
        self.I_d = -Q_anode * cfg.A_channel / cfg.dt
        # Katoda dosyła elektrony, mniej więcej tyle, ile jonów do niej dotarło.
        inj_w = cfg.cathode_gain * n_i_cath
        pusher.inject_cathode_electrons(self.electrons, self.w_ref, inj_w, cfg, self.rng)
        # Zderzenia z gazem.
        if cfg.enable_collisions:
            self.n_ionized = self.mcc.collide_electrons(self.electrons, self.ions, self.w_ref)
            self.mcc.collide_ions(self.ions, self.w_ref)
            self.total_ionized += self.n_ionized
        # Co pewien czas dzielimy i łączymy czastki.
        if cfg.enable_apr and (self.step % cfg.apr_interval == 0):
            self.n_split, self.n_merge = apr_mod.run_apr(self.electrons, cfg, self.w_ref, self.rng)
            self.total_split += self.n_split
            self.total_merge += self.n_merge
        # Przesunięcie obwodu.
        self.circuit.advance(self.I_d, cfg.dt)
        # Miara prądu niesionego przez rozpędzone elektrony.
        self.I_RE = self._beam_current()

        self.t += cfg.dt
        self.step += 1

    def _beam_current(self):
        """Szacuje prąd niesiony przez rozpędzone elektrony.

        Bierzemy tylko elektrony powyżej progu i sumujemy ich udział w ruchu
        ładunku wzdłuż osi.
        """
        el = self.electrons
        if el.N == 0:
            return 0.0
        eps = el.kinetic_energy_eV()
        mask = eps > self.cfg.E_RE_eV
        if not np.any(mask):
            return 0.0
        # Średni przepływ ładunku niesiony przez rozpędzone elektrony.
        flux = np.sum(el.aw[mask] * el.avx[mask]) / self.cfg.L
        return abs(E_CHARGE * self.cfg.A_channel * flux)

    # Tekst do panelu diagnostycznego.
    def stats_text(self):
        """Składa wielowierszowy opis bieżącego stanu do pokazania w podglądzie."""
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
