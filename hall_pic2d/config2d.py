"""Wszystkie nastawy wersji dwuwymiarowej, z wyborem płaszczyzny kanału.

Można pracować na jednej z dwóch płaszczyzn. Pierwsza obejmuje oś kanału i
kierunek wokół niego. Kierunek wokół kanału zawija się sam w sobie, bo
biegnie po okręgu, a pole magnetyczne wychodzi wtedy prostopadle poza
płaszczyznę. Ta płaszczyzna dobrze pokazuje niestabilność dryfu elektronów.
Druga płaszczyzna obejmuje oś kanału i kierunek w poprzek, od jednej ścianki
do drugiej. Ścianki pochłaniają czastki, a pole magnetyczne leży wtedy w
płaszczyźnie. Ta z kolei pokazuje straty na ściankach.

Prędkość ma zawsze trzy składowe: dwie w płaszczyźnie obliczeń i jedną poza
nią. Wagę czastki liczymy tak, aby po przeskalowaniu na cały obwód silnika
prąd wychodził w tej samej skali w obu płaszczyznach; służy do tego osobny
przelicznik dobierany zależnie od wybranej płaszczyzny.
"""

from dataclasses import dataclass
import numpy as np

from hall_pic.constants import M_XENON


@dataclass
class Config2D:
    """Konfiguracja wersji dwuwymiarowej z wyborem płaszczyzny kanału."""

    geometry: str = "z-theta"        # która płaszczyzna: wokół kanału albo w poprzek

    # Wymiary wspólne dla obu płaszczyzn.
    L1: float = 0.025                # długość kanału wzdłuż osi
    R_mean: float = 0.0425           # średni promień kanału
    channel_width: float = 0.015     # szerokość kanału między ściankami
    L2_theta: float = 0.010          # jak szeroki wycinek wokół kanału liczymy

    N1: int = 128                    # na tyle komórek dzielimy oś kanału
    N2: int = 64                     # na tyle komórek dzielimy drugi kierunek

    # Pole magnetyczne, tak jak w wersji jednowymiarowej: dzwon ze szczytem przy wylocie.
    B_max: float = 0.018             # natężenie w szczycie
    x_B: float = 0.020               # gdzie na osi wypada szczyt
    sigma_B: float = 0.004           # jak szeroki jest szczyt

    # Zamrożony gaz obojętny.
    n_n_anode: float = 3.0e18        # gęstość przy anodzie
    n_n_exit: float = 3.0e17         # gęstość przy wylocie
    T_neutral: float = 500.0         # temperatura gazu
    m_ion: float = M_XENON

    # Obwód zasilający. Nazwy pól są takie same jak w wersji jednowymiarowej,
    # żeby dało się użyć tego samego modelu obwodu.
    V_ps: float = 400.0
    R_circ: float = 10.0
    L_circ: float = 1.0e-4
    C_circ: float = 1.0e-7
    V_C_init: float = 350.0
    I_L_init: float = 4.0
    circuit_substeps: int = 10

    # Plazma początkowa.
    n0_plasma: float = 5.0e16
    Te0_eV: float = 3.0
    Ti0_eV: float = 0.5
    n_ppc: int = 20                  # ile czastek modelowych na komórkę i gatunek
    Te_cathode_eV: float = 3.0
    cathode_gain: float = 1.0

    # Zderzenia.
    enable_collisions: bool = True
    eps_max_grid_eV: float = 1000.0
    n_energy_grid: int = 2000

    # Dzielenie i łączenie czastek.
    enable_apr: bool = True
    apr_interval: int = 20
    E_RE_eV: float = 50.0
    apr_beam_frac_threshold: float = 0.03
    apr_min_weight_ratio: float = 0.02
    # Progi dzielenia i łączenia zapisujemy jako krotność liczby czastek na
    # komórkę, a nie jako sztywne liczby. Gdyby były sztywne, zmniejszenie
    # liczby czastek na starcie ustawiłoby cel wielokrotnie wyżej niż punkt
    # wyjścia i czastek przybywałoby lawinowo, spowalniając krok.
    apr_split_target_factor: float = 4.0   # do ilu razy więcej czastek dążymy w wiązce
    apr_max_ppc_factor: float = 2.0        # ile razy więcej czastek w tle wyzwala łączenie
    max_particles: int = 1_500_000         # twardy sufit, powyżej którego nie dzielimy

    # Krok i długość przebiegu.
    dt: float = 5.0e-12
    t_end: float = 5.0e-8

    # Podgląd i zapis.
    live_view: bool = True
    plot_interval: int = 50
    headless_save_dir: str = ""
    max_scatter_points: int = 6000

    seed: int = 12345

    def __post_init__(self):
        """Dolicza to, co zależy od wybranej płaszczyzny, i sprawdza jej nazwę."""
        if self.geometry not in ("z-theta", "z-r"):
            raise ValueError("geometry musi być 'z-theta' albo 'z-r'")

        circumference = 2.0 * np.pi * self.R_mean

        if self.geometry == "z-theta":
            self.L2 = self.L2_theta
            self.x2_bc = "periodic"
            self.x2_label = "θ·R [mm]"
            # Pomijamy tu kierunek w poprzek kanału, więc prąd z liczonego
            # wycinka przeskalowujemy na pełny obwód silnika.
            self.current_factor = self.channel_width * (circumference / self.L2)
        else:  # płaszczyzna w poprzek kanału
            self.L2 = self.channel_width
            self.x2_bc = "dirichlet"
            self.x2_label = "r - r_in [mm]"
            # Pomijamy tu kierunek wokół kanału, więc prąd przeskalowujemy na
            # cały obwód.
            self.current_factor = circumference

        self.h1 = self.L1 / self.N1
        self.h2 = self.L2 / self.N2
        self.n1_nodes = self.N1 + 1
        # Przy kierunku zawijającym się w sobie ostatni węzeł pokrywa się z
        # pierwszym, więc jest ich o jeden mniej.
        self.n2_nodes = self.N2 if self.x2_bc == "periodic" else self.N2 + 1

        self.x1_nodes = np.linspace(0.0, self.L1, self.n1_nodes)
        if self.x2_bc == "periodic":
            self.x2_nodes = np.arange(self.N2) * self.h2
        else:
            self.x2_nodes = np.linspace(0.0, self.L2, self.n2_nodes)

        self.n_cells = self.N1 * self.N2
        self.n_steps = int(round(self.t_end / self.dt))

        # Progi dzielenia i łączenia przeliczone na konkretne liczby czastek.
        self.apr_split_target_ppc = max(1, int(self.apr_split_target_factor * self.n_ppc))
        self.apr_max_ppc = max(1, int(self.apr_max_ppc_factor * self.n_ppc))

    # Wielkości fizyczne zależne od miejsca na osi.
    def B_profile(self, z):
        """Zwraca natężenie pola magnetycznego w zadanym miejscu na osi kanału."""
        return self.B_max * np.exp(-0.5 * ((z - self.x_B) / self.sigma_B) ** 2)

    def B_vector(self, z):
        """Zwraca pole magnetyczne jako trzy składowe, ustawione zależnie od płaszczyzny.

        Na jednej płaszczyźnie pole wychodzi prostopadle poza nią, na drugiej
        leży w niej, wzdłuż kierunku w poprzek kanału.
        """
        B = self.B_profile(z)
        zeros = np.zeros_like(B)
        if self.geometry == "z-theta":
            return zeros, zeros, B      # pole wychodzi poza płaszczyznę
        return zeros, B, zeros          # pole leży w płaszczyźnie

    def neutral_density(self, z):
        """Zwraca gęstość zamrożonego gazu, zależną tylko od miejsca na osi kanału."""
        frac = np.clip(z / self.L1, 0.0, 1.0)
        return self.n_n_anode * (1.0 - frac) + self.n_n_exit * frac
