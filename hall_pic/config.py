"""Wszystkie nastawy symulacji zebrane w jednym miejscu.

Domyślne wartości odpowiadają silnikowi zbliżonemu do SPT-100: kanał ma około
dwóch i pół centymetra, a pole magnetyczne osiąga maksimum mniej więcej sto
osiemdziesiąt gausów tuż przy wylocie. Liczby trzymamy w jednostkach układu
SI, chyba że w komentarzu napisano inaczej. Parametry pogrupowano tematycznie,
żeby łatwo było znaleźć to, co chcemy zmienić.
"""

from dataclasses import dataclass, field
import numpy as np

from .constants import M_XENON


@dataclass
class Config:
    """Konfiguracja symulacji silnika Halla w wersji jednowymiarowej."""

    # Geometria kanału. Anoda leży na lewym końcu, katoda na prawym.
    L: float = 0.025                # długość kanału wzdłuż osi, w metrach
    Nx: int = 256                   # na tyle komórek dzielimy kanał
    A_channel: float = 4.0e-3       # pole przekroju kanału, potrzebne do prądu

    # Pole magnetyczne rośnie i opada łagodnie, z jednym wyraźnym szczytem
    # ulokowanym blisko wylotu. Taki kształt zatrzymuje elektrony.
    B_max: float = 0.018            # jak silne jest pole w szczycie
    x_B: float = 0.020              # gdzie na osi wypada ten szczyt
    sigma_B: float = 0.004          # jak szeroki jest szczyt

    # Gaz obojętny jest zamrożony: jego rozkład jest z góry ustalony i nie
    # zmienia się w czasie. Celowo rozrzedziliśmy go mniej więcej dziesięć razy
    # w stosunku do wartości typowej, bo przy rzadszym gazie elektrony rzadziej
    # się zderzają i łatwiej rozpędzają, a właśnie te rozpędzone chcemy widzieć.
    n_n_anode: float = 3.0e18       # gęstość gazu przy anodzie
    n_n_exit: float = 3.0e17        # gęstość gazu przy wylocie
    T_neutral: float = 500.0        # temperatura gazu, w kelwinach
    m_ion: float = M_XENON          # masa pojedynczego jonu

    # Zewnętrzny obwód zasilający. Zasilacz podaje napięcie przez opór i cewkę,
    # a równolegle do wyładowania wisi kondensator, który wygładza przebieg.
    # Napięcie na tym kondensatorze jest zarazem napięciem anody.
    #
    # Indukcyjność jest celowo mała, żeby obwód zdążył zareagować w trakcie
    # przebiegu. Prąd w gałęzi z cewką startuje od zera, więc docelowe kilka
    # amperów nie jest wpisane z góry, tylko samo ustala się z równowagi między
    # zasilaczem, oporem balastowym a prądem, jaki pobiera wyładowanie.
    V_ps: float = 400.0             # napięcie zasilacza; wyższe daje więcej rozpędzonych elektronów
    R_circ: float = 10.0            # opór szeregowy, pełni rolę balastu i ustala prąd roboczy
    L_circ: float = 1.0e-6          # indukcyjność szeregowa, mała, żeby obwód nadążał za przebiegiem
    C_circ: float = 1.0e-7          # pojemność wygładzająca
    V_C_init: float = 300.0         # od jakiego napięcia anody zaczynamy
    I_L_init: float = 0.0           # prąd w gałęzi z cewką startuje od zera i sam narasta

    # Prąd wyładowania odczytany z pojedynczego kroku jest bardzo poszarpany,
    # bo każda czastka trafiająca na anodę wnosi od razu spory ładunek. Zanim
    # podamy go do obwodu, uśredniamy go po oknie obejmującym o wiele więcej
    # niż jeden przelot czastki przez kanał, dzięki czemu obwód widzi gładki
    # prąd, a nie pojedyncze skoki.
    I_d_avg_window: int = 4000      # po ilu krokach uśredniamy prąd wyładowania

    # Plazma, od której startujemy, oraz liczba czastek modelowych.
    n0_plasma: float = 5.0e16       # początkowa gęstość plazmy, w przybliżeniu obojętna elektrycznie
    Te0_eV: float = 3.0             # początkowa temperatura elektronów; chłodne tło uwypukla wiązkę
    Ti0_eV: float = 0.5             # początkowa temperatura jonów
    n_ppc: int = 120                # ile czastek modelowych na komórkę i gatunek zakładamy na starcie

    # Katoda po prawej stronie dosyła elektrony w głąb kanału, tak jak robi to
    # neutralizator w prawdziwym silniku.
    Te_cathode_eV: float = 3.0      # temperatura dosyłanych elektronów
    cathode_gain: float = 1.0       # o ile emisja przewyższa strumień jonów docierających do katody

    # Zderzenia liczymy metodą zderzeń zerowych. Gaz jest zamrożony.
    enable_collisions: bool = True
    eps_max_grid_eV: float = 1000.0 # do jakiej energii sięgamy, szukając największej częstości zderzeń
    n_energy_grid: int = 2000       # jak gęsto próbkujemy energię przy tym szukaniu

    # Adaptacyjne dzielenie czastek. Ma poprawić statystykę tam, gdzie tworzy
    # się wiązka rozpędzonych elektronów.
    enable_apr: bool = True
    apr_interval: int = 20          # co ile kroków uruchamiamy tę procedurę
    E_RE_eV: float = 50.0           # od jakiej energii uznajemy elektron za rozpędzony
    apr_beam_frac_threshold: float = 0.03  # jaki udział rozpędzonych czyni z komórki obszar wiązki
    apr_split_target_ppc: int = 500 # do ilu czastek na komórkę dążymy w obszarze wiązki
    apr_max_ppc: int = 250          # powyżej tylu czastek w tle zaczynamy je łączyć
    apr_min_weight_ratio: float = 0.02  # nie dzielimy czastek lżejszych niż ten ułamek wagi wzorcowej

    # Krok czasowy i długość przebiegu.
    dt: float = 5.0e-12             # krok czasowy; dobrany tak, by nadążać za najszybszymi drganiami plazmy
    t_end: float = 2.0e-7           # dokąd liczymy; można wydłużyć do jednej mikrosekundy
    circuit_substeps: int = 10      # na tyle mniejszych kroków dzielimy całkowanie obwodu wewnątrz jednego kroku

    # Podgląd i zapis wyników.
    live_view: bool = True          # czy pokazywać okno z wykresami na żywo
    plot_interval: int = 50         # co ile kroków odświeżamy wykresy
    headless_save_dir: str = ""     # jeśli podamy katalog, zamiast okna zapisujemy klatki do plików
    max_scatter_points: int = 6000  # ile najwyżej punktów rysujemy na wykresie przestrzeni fazowej

    seed: int = 12345               # ziarno losowości, żeby przebieg dało się powtórzyć

    def __post_init__(self):
        """Dolicza to, co wynika wprost z podanych nastaw: siatkę i liczbę kroków."""
        self.dx = self.L / self.Nx
        self.x_nodes = np.linspace(0.0, self.L, self.Nx + 1)
        self.x_cells = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])
        self.n_nodes = self.Nx + 1
        self.n_steps = int(round(self.t_end / self.dt))

    def B_profile(self, x):
        """Zwraca natężenie pola magnetycznego w zadanym miejscu na osi.

        Kształt jest dzwonowy, z jednym szczytem przy wylocie kanału.
        """
        return self.B_max * np.exp(-0.5 * ((x - self.x_B) / self.sigma_B) ** 2)

    def neutral_density(self, x):
        """Zwraca gęstość zamrożonego gazu w zadanym miejscu na osi.

        Gaz jest najgęstszy przy anodzie i liniowo rzednie w stronę wylotu.
        """
        frac = np.clip(x / self.L, 0.0, 1.0)
        return self.n_n_anode * (1.0 - frac) + self.n_n_exit * frac
