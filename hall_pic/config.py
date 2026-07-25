"""Konfiguracja symulacji PIC 1D3V silnika Halla (geometria SPT-100).

Wszystkie wielkości w jednostkach SI, o ile nie zaznaczono inaczej.
Parametry są pogrupowane tematycznie i mają rozsądne wartości domyślne
dla benchmarku typu SPT-100 (kanał osiowy ~2.5 cm, B_r ~180 G przy wylocie).
"""

from dataclasses import dataclass, field
import numpy as np

from .constants import M_XENON


@dataclass
class Config:
    # ---------------------------------------------------------------
    # Geometria (osiowa, 1D). x = 0 -> anoda, x = L -> płaszczyzna katody.
    # ---------------------------------------------------------------
    L: float = 0.025                # długość domeny osiowej [m]
    Nx: int = 256                   # liczba komórek siatki
    A_channel: float = 4.0e-3       # pole przekroju kanału [m^2] (do prądu obwodu)

    # Profil radialnego pola magnetycznego B_r(x) = B_max * exp(-(x-x_B)^2 / 2 sigma^2)
    B_max: float = 0.018            # szczyt indukcji [T] (~180 G)
    x_B: float = 0.020              # położenie szczytu B (blisko wylotu) [m]
    sigma_B: float = 0.004          # szerokość gaussa profilu B [m]

    # ---------------------------------------------------------------
    # Neutrale — ZAMROŻONE (stały profil gęstości, brak dynamiki)
    # ---------------------------------------------------------------
    # Obniżone ~10x względem wartości bazowej: mniejsza kolizyjność ->
    # elektrony łatwiej "uciekają" w energii (wyraźniejsza wiązka RE).
    n_n_anode: float = 3.0e18       # gęstość neutrali przy anodzie [1/m^3]
    n_n_exit: float = 3.0e17        # gęstość neutrali przy wylocie [1/m^3]
    T_neutral: float = 500.0        # temperatura neutrali [K]
    m_ion: float = M_XENON          # masa jonu [kg]

    # ---------------------------------------------------------------
    # Obwód zewnętrzny RLC z kondensatorem (samouzgodniony)
    #   V_ps --[R]--[L_circ]--+---(anoda | katoda)   ; C równolegle do wyładowania
    #                         |
    #                        [C]
    # ---------------------------------------------------------------
    V_ps: float = 400.0             # napięcie zasilacza [V] (silniejsze pole -> więcej RE)
    R_circ: float = 10.0            # rezystancja szeregowa (balast) [Ohm]
    L_circ: float = 1.0e-4          # indukcyjność szeregowa [H]
    C_circ: float = 1.0e-7          # pojemność filtrująca [F]
    V_C_init: float = 350.0         # początkowe napięcie na kondensatorze/anodzie [V]
    I_L_init: float = 4.0           # początkowy prąd w gałęzi indukcyjnej [A] (~cel 4 A)

    # ---------------------------------------------------------------
    # Plazma początkowa i cząstki makro
    # ---------------------------------------------------------------
    n0_plasma: float = 5.0e16       # początkowa gęstość plazmy (quasineutralna) [1/m^3]
    Te0_eV: float = 3.0             # początkowa temp. elektronów [eV] (chłodne tło -> wiązka wyraźniejsza)
    Ti0_eV: float = 0.5             # początkowa temp. jonów [eV]
    n_ppc: int = 120                # cząstek makro na komórkę (na gatunek) na starcie

    # Emisja z katody (neutralizator) — elektrony wstrzykiwane w x=L
    Te_cathode_eV: float = 3.0      # temperatura wstrzykiwanych elektronów [eV]
    cathode_gain: float = 1.0       # mnożnik prądu emisji względem strumienia jonów do katody

    # ---------------------------------------------------------------
    # Kolizje (null-collision MCC), neutrale zamrożone
    # ---------------------------------------------------------------
    enable_collisions: bool = True
    eps_max_grid_eV: float = 1000.0 # górna energia do wyznaczenia nu_max
    n_energy_grid: int = 2000       # rozdzielczość siatki energii dla nu_max

    # ---------------------------------------------------------------
    # Adaptive Particle Refinement (APR) dla wiązek RE
    # ---------------------------------------------------------------
    enable_apr: bool = True
    apr_interval: int = 20          # co ile kroków uruchamiać APR
    E_RE_eV: float = 50.0           # próg energii "runaway" definiujący wiązkę RE [eV]
    apr_beam_frac_threshold: float = 0.03  # frakcja el. RE w komórce -> komórka "wiązkowa"
    apr_split_target_ppc: int = 500 # docelowa liczba el. na komórkę w obszarze wiązki
    apr_max_ppc: int = 250          # limit el. na komórkę w obszarze zbiorczym (merge)
    apr_min_weight_ratio: float = 0.02  # nie dziel poniżej tej frakcji wagi referencyjnej

    # ---------------------------------------------------------------
    # Krok czasowy i czas symulacji
    # ---------------------------------------------------------------
    dt: float = 5.0e-12             # krok czasowy [s] (rozdziela omega_pe i omega_ce)
    t_end: float = 2.0e-7           # czas końcowy [s] (200 ns; dociągnij do 1e-6 dla pełnego)
    circuit_substeps: int = 10      # podkroki całkowania ODE obwodu na 1 krok PIC

    # ---------------------------------------------------------------
    # Diagnostyka / podgląd na żywo
    # ---------------------------------------------------------------
    live_view: bool = True          # okno Matplotlib na żywo
    plot_interval: int = 50         # co ile kroków odświeżać wykresy
    headless_save_dir: str = ""     # jeśli != "" zapisuj klatki PNG zamiast okna
    max_scatter_points: int = 6000  # limit punktów w wykresie przestrzeni fazowej

    seed: int = 12345               # ziarno RNG (powtarzalność)

    # ---------------------------------------------------------------
    # Wielkości pochodne (wyliczane po inicjalizacji)
    # ---------------------------------------------------------------
    def __post_init__(self):
        self.dx = self.L / self.Nx
        self.x_nodes = np.linspace(0.0, self.L, self.Nx + 1)
        self.x_cells = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])
        self.n_nodes = self.Nx + 1
        # liczba kroków
        self.n_steps = int(round(self.t_end / self.dt))

    # Profil pola magnetycznego na węzłach
    def B_profile(self, x):
        return self.B_max * np.exp(-0.5 * ((x - self.x_B) / self.sigma_B) ** 2)

    # Profil gęstości neutrali (liniowy spadek anoda->wylot), zamrożony
    def neutral_density(self, x):
        frac = np.clip(x / self.L, 0.0, 1.0)
        return self.n_n_anode * (1.0 - frac) + self.n_n_exit * frac
