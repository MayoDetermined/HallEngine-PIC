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
        self.I_d_avg = 0.0
        self.I_RE = 0.0
        self.n_split = 0
        self.n_merge = 0
        self.n_ionized = 0
        # Liczniki zbierane od początku przebiegu.
        self.total_split = 0
        self.total_merge = 0
        self.total_ionized = 0

        # Bufor kołowy do uśredniania prądu wyładowania podawanego do obwodu.
        self._Id_buf = np.zeros(cfg.I_d_avg_window)
        self._Id_pos = 0
        self._Id_count = 0

        # Bilans energii. Liczymy energię kinetyczną obu gatunków w dżulach i
        # zbieramy, ile energii wnosi pole, a ile ubywa na ściankach, w
        # zderzeniach i przy dosyłaniu z katody. Osobno pilnujemy, ile energii
        # zmienia dzielenie i łączenie czastek, bo powinno ono energię zachować.
        self.E_field = 0.0      # praca pola nad czastkami
        self.E_wall = 0.0       # energia uniesiona przez czastki na elektrody
        self.E_coll = 0.0       # zmiana energii w zderzeniach z gazem
        self.E_cath = 0.0       # energia wniesiona przez elektrony z katody
        self.E_apr = 0.0        # zmiana energii przy dzieleniu i łączeniu (kontrola)

        self._seed_plasma()
        self.KE0 = self._kinetic_energy_J()

    def _kinetic_energy_J(self):
        """Łączna energia kinetyczna obu gatunków, w dżulach."""
        ke = 0.0
        for sp in (self.electrons, self.ions):
            if sp.N:
                ke += 0.5 * sp.mass * np.sum(sp.aw * sp.speed2())
        return ke * self.cfg.A_channel

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

        # Energia kinetyczna przed pchnięciem, do bilansu energii.
        ke_pre = self._kinetic_energy_J()
        # Odczyt pola w miejscu czastek i ich przesunięcie.
        Ee = pusher.gather_field(self.electrons, self.E, cfg)
        Ei = pusher.gather_field(self.ions, self.E, cfg)
        pusher.boris_push(self.electrons, Ee, cfg)
        pusher.boris_push(self.ions, Ei, cfg)
        # Pchacz zmienia energię tylko przez pole elektryczne, bo obrót w polu
        # magnetycznym nie zmienia szybkości. Różnica energii to praca pola.
        ke_push = self._kinetic_energy_J()
        self.E_field += ke_push - ke_pre

        # Pochłonięcie czastek na elektrodach i odczyt prądu wyładowania.
        qa_e, qc_e, _ = pusher.apply_boundaries(self.electrons, cfg)
        qa_i, qc_i, n_i_cath = pusher.apply_boundaries(self.ions, cfg)
        # Energia, którą uniosły pochłonięte czastki.
        ke_bound = self._kinetic_energy_J()
        self.E_wall += ke_push - ke_bound
        # Ładunek zebrany na anodzie zamieniamy na prąd. Znak wynika z tego,
        # że ładunek elektronu jest ujemny.
        Q_anode = qa_e + qa_i
        self.I_d = -Q_anode * cfg.A_channel / cfg.dt

        # Katoda dosyła elektrony, mniej więcej tyle, ile jonów do niej dotarło.
        inj_w = cfg.cathode_gain * n_i_cath
        pusher.inject_cathode_electrons(self.electrons, self.w_ref, inj_w, cfg, self.rng)
        ke_inj = self._kinetic_energy_J()
        self.E_cath += ke_inj - ke_bound

        # Zderzenia z gazem.
        if cfg.enable_collisions:
            self.n_ionized = self.mcc.collide_electrons(self.electrons, self.ions, self.w_ref)
            self.mcc.collide_ions(self.ions, self.w_ref)
            self.total_ionized += self.n_ionized
        ke_coll = self._kinetic_energy_J()
        self.E_coll += ke_coll - ke_inj

        # Co pewien czas dzielimy i łączymy czastki. Ta operacja powinna
        # zachować energię, więc jej wkład do bilansu ma być bliski zeru.
        if cfg.enable_apr and (self.step % cfg.apr_interval == 0):
            self.n_split, self.n_merge = apr_mod.run_apr(self.electrons, cfg, self.w_ref, self.rng)
            self.total_split += self.n_split
            self.total_merge += self.n_merge
        ke_apr = self._kinetic_energy_J()
        self.E_apr += ke_apr - ke_coll

        # Prąd wyładowania podawany do obwodu uśredniamy po oknie, żeby obwód
        # widział gładki przebieg zamiast pojedynczych skoków.
        self._Id_buf[self._Id_pos] = self.I_d
        self._Id_pos = (self._Id_pos + 1) % self._Id_buf.size
        self._Id_count = min(self._Id_count + 1, self._Id_buf.size)
        self.I_d_avg = float(self._Id_buf[:self._Id_count].mean())

        # Przesunięcie obwodu, sterowane uśrednionym prądem.
        self.circuit.advance(self.I_d_avg, cfg.dt)
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

    def energy_balance(self):
        """Zwraca składniki bilansu energii oraz resztę zamknięcia.

        Zmiana energii kinetycznej od początku przebiegu powinna równać się
        pracy pola, pomniejszonej o straty na ściankach i w zderzeniach, a
        powiększonej o energię z katody. Wkład dzielenia i łączenia czastek
        powinien być bliski zeru. Reszta pokazuje, jak dobrze bilans się domyka.
        """
        ke_now = self._kinetic_energy_J()
        expected = (self.E_field - self.E_wall + self.E_cath
                    + self.E_coll + self.E_apr)
        residual = (ke_now - self.KE0) - expected
        throughput = max(abs(self.E_field), 1e-30)
        return {
            "KE": ke_now,
            "dKE": ke_now - self.KE0,
            "field": self.E_field,
            "wall": self.E_wall,
            "coll": self.E_coll,
            "cath": self.E_cath,
            "apr": self.E_apr,
            "residual": residual,
            "residual_rel": residual / throughput,
        }

    # Tekst do panelu diagnostycznego.
    def stats_text(self):
        """Składa wielowierszowy opis bieżącego stanu do pokazania w podglądzie."""
        el = self.electrons
        eps = el.kinetic_energy_eV() if el.N > 0 else np.array([0.0])
        n_re = int(np.sum(eps > self.cfg.E_RE_eV)) if el.N > 0 else 0
        mean_eps = float(np.average(eps, weights=el.aw)) if el.N > 0 else 0.0
        max_eps = float(np.max(eps)) if el.N > 0 else 0.0
        eb = self.energy_balance()
        return (
            f"krok         : {self.step}/{self.cfg.n_steps}\n"
            f"t            : {self.t*1e9:8.2f} ns\n"
            f"N_e (makro)  : {el.N}\n"
            f"N_i (makro)  : {self.ions.N}\n"
            f"---- obwód ----\n"
            f"I_d (chwil.) : {self.I_d:8.3f} A\n"
            f"I_d (uśred.) : {self.I_d_avg:8.3f} A\n"
            f"I_L          : {self.circuit.I_L:8.3f} A\n"
            f"V_C (anoda)  : {self.circuit.V_C:8.2f} V\n"
            f"---- wiązka RE ----\n"
            f"E_RE próg    : {self.cfg.E_RE_eV:6.1f} eV\n"
            f"N_e(RE)      : {n_re}\n"
            f"I_RE         : {self.I_RE:8.4f} A\n"
            f"<E_e>        : {mean_eps:8.2f} eV\n"
            f"max E_e      : {max_eps:8.1f} eV\n"
            f"---- bilans energii [J] ----\n"
            f"zmiana E_kin : {eb['dKE']:9.3e}\n"
            f"praca pola   : {eb['field']:9.3e}\n"
            f"straty ścian : {eb['wall']:9.3e}\n"
            f"zderzenia    : {eb['coll']:9.3e}\n"
            f"reszta APR   : {eb['apr']:9.3e}\n"
            f"reszta bil.  : {eb['residual_rel']:9.2e}\n"
            f"---- APR/MCC (skumul.) ----\n"
            f"split/merge  : {self.total_split} / {self.total_merge}\n"
            f"jonizacje    : {self.total_ionized}\n"
        )
