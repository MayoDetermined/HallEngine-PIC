"""Konfiguracja symulacji PIC 2D3V silnika Halla, dwie przełączalne geometrie.

GEOMETRIE
---------
'z-theta'  — płaszczyzna osiowo-azymutalna (standard typu LANDMARK)
    x1 = z (osiowa, anoda/katoda), x2 = θ·R (azymutalna, PERIODYCZNA)
    B radialne = PROSTOPADŁE do płaszczyzny -> B = (0, 0, B_r(z))
    Dryf ExB leży W płaszczyźnie -> chwyta niestabilność dryfu ExB.

'z-r'      — płaszczyzna osiowo-promieniowa (kanał ze ściankami)
    x1 = z (osiowa), x2 = r − r_in (promieniowa, ŚCIANKI pochłaniające)
    B radialne leży W płaszczyźnie -> B = (0, B_r(z), 0)
    Dryf ExB jest PROSTOPADŁY do płaszczyzny (3. składowa prędkości).

Konwencja prędkości: v1 wzdłuż x1, v2 wzdłuż x2, v3 poza płaszczyzną.

WAŻENIE (2D): waga superczątstki w ma jednostkę [1/m] — liczba cząstek
rzeczywistych NA METR kierunku ignorowanego (3. wymiar). Wtedy:
    gęstość: n = Σ w·S(x−x_p) / (h1·h2)                        [1/m³]
    prąd   : I = q · Σw · current_factor / dt                  [A]
gdzie current_factor [m] przelicza symulowany wycinek na pełny silnik.
"""

from dataclasses import dataclass
import numpy as np

from hall_pic.constants import M_XENON


@dataclass
class Config2D:
    # --------------------- wybór geometrii ---------------------
    geometry: str = "z-theta"        # 'z-theta' albo 'z-r'

    # --------------------- geometria wspólna ---------------------
    L1: float = 0.025                # długość osiowa (z) [m]
    R_mean: float = 0.0425           # średni promień kanału [m]
    channel_width: float = 0.015     # szerokość kanału (r_out − r_in) [m]
    L2_theta: float = 0.010          # symulowany wycinek azymutalny [m] (periodyczny)

    N1: int = 128                    # komórki wzdłuż z
    N2: int = 64                     # komórki wzdłuż x2

    # Profil pola magnetycznego B_r(z) — gaussowski przy wylocie
    B_max: float = 0.018             # [T] (~180 G)
    x_B: float = 0.020               # położenie szczytu [m]
    sigma_B: float = 0.004           # szerokość [m]

    # --------------------- neutrale (ZAMROŻONE) ---------------------
    n_n_anode: float = 3.0e18        # [1/m³]
    n_n_exit: float = 3.0e17         # [1/m³]
    T_neutral: float = 500.0         # [K]
    m_ion: float = M_XENON

    # --------------------- obwód RLC (nazwy zgodne z hall_pic.circuit) ---------
    V_ps: float = 400.0
    R_circ: float = 10.0
    L_circ: float = 1.0e-4
    C_circ: float = 1.0e-7
    V_C_init: float = 350.0
    I_L_init: float = 4.0
    circuit_substeps: int = 10

    # --------------------- plazma początkowa ---------------------
    n0_plasma: float = 5.0e16        # [1/m³]
    Te0_eV: float = 3.0
    Ti0_eV: float = 0.5
    n_ppc: int = 20                  # cząstek makro na komórkę na gatunek
    Te_cathode_eV: float = 3.0
    cathode_gain: float = 1.0

    # --------------------- kolizje ---------------------
    enable_collisions: bool = True
    eps_max_grid_eV: float = 1000.0
    n_energy_grid: int = 2000

    # --------------------- APR ---------------------
    enable_apr: bool = True
    apr_interval: int = 20
    E_RE_eV: float = 50.0
    apr_beam_frac_threshold: float = 0.03
    apr_min_weight_ratio: float = 0.02
    # Progi APR są WZGLĘDNE wobec n_ppc — wartości bezwzględne przy zmianie
    # --ppc dawałyby cel wielokrotnie wyższy od startu i lawinowy wzrost
    # liczby cząstek (a więc spowolnienie kroku).
    apr_split_target_factor: float = 4.0   # cel w komórkach wiązkowych = 4·n_ppc
    apr_max_ppc_factor: float = 2.0        # próg merge w tle          = 2·n_ppc
    max_particles: int = 1_500_000         # twardy limit: powyżej nie dzielimy

    # --------------------- czas ---------------------
    dt: float = 5.0e-12
    t_end: float = 5.0e-8

    # --------------------- diagnostyka ---------------------
    live_view: bool = True
    plot_interval: int = 50
    headless_save_dir: str = ""
    max_scatter_points: int = 6000

    seed: int = 12345

    # --------------------- wielkości pochodne ---------------------
    def __post_init__(self):
        if self.geometry not in ("z-theta", "z-r"):
            raise ValueError("geometry musi być 'z-theta' albo 'z-r'")

        circumference = 2.0 * np.pi * self.R_mean

        if self.geometry == "z-theta":
            self.L2 = self.L2_theta
            self.x2_bc = "periodic"
            self.x2_label = "θ·R [mm]"
            # kierunek ignorowany = promieniowy (szerokość kanału);
            # wycinek azymutalny skalujemy do pełnego obwodu
            self.current_factor = self.channel_width * (circumference / self.L2)
        else:  # 'z-r'
            self.L2 = self.channel_width
            self.x2_bc = "dirichlet"
            self.x2_label = "r − r_in [mm]"
            # kierunek ignorowany = azymut (pełny obwód)
            self.current_factor = circumference

        self.h1 = self.L1 / self.N1
        self.h2 = self.L2 / self.N2
        self.n1_nodes = self.N1 + 1
        # periodyczny: węzeł N2 ≡ 0, więc jest ich N2
        self.n2_nodes = self.N2 if self.x2_bc == "periodic" else self.N2 + 1

        self.x1_nodes = np.linspace(0.0, self.L1, self.n1_nodes)
        if self.x2_bc == "periodic":
            self.x2_nodes = np.arange(self.N2) * self.h2
        else:
            self.x2_nodes = np.linspace(0.0, self.L2, self.n2_nodes)

        self.n_cells = self.N1 * self.N2
        self.n_steps = int(round(self.t_end / self.dt))

        # progi APR skalowane z n_ppc (patrz uwaga przy definicji pól)
        self.apr_split_target_ppc = max(1, int(self.apr_split_target_factor * self.n_ppc))
        self.apr_max_ppc = max(1, int(self.apr_max_ppc_factor * self.n_ppc))

    # ---- profile fizyczne ----
    def B_profile(self, z):
        """Moduł B_r w funkcji położenia osiowego."""
        return self.B_max * np.exp(-0.5 * ((z - self.x_B) / self.sigma_B) ** 2)

    def B_vector(self, z):
        """B jako 3-wektor (b1, b2, b3) zgodnie z geometrią."""
        B = self.B_profile(z)
        zeros = np.zeros_like(B)
        if self.geometry == "z-theta":
            return zeros, zeros, B      # prostopadłe do płaszczyzny
        return zeros, B, zeros          # w płaszczyźnie (wzdłuż r)

    def neutral_density(self, z):
        """Zamrożony profil neutrali (zależny tylko od osi z)."""
        frac = np.clip(z / self.L1, 0.0, 1.0)
        return self.n_n_anode * (1.0 - frac) + self.n_n_exit * frac
