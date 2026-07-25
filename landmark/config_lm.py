"""Warunki benchmarku LANDMARK 1D-axial (Hagelaar, Hara, Smolyakov, Boeuf, 2018).

Odwzorowanie specyfikacji 1:1, żeby konfiguracja była weryfikowalna i wielokrotnego
użytku. Wartości wprost z dokumentu "Model-1D-HALL Benchmark".

UWAGA — LANDMARK jest benchmarkiem modeli PŁYNOWYCH / hybrydowych z założeniem
QUASINEUTRALNOŚCI i NARZUCONYM transportem anomalnym. Nie jest benchmarkiem PIC.
Różnice względem tego kodu opisano w `README_LANDMARK.md`.
"""

from dataclasses import dataclass
import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON, AMU


@dataclass
class LandmarkConfig:
    # --- geometria ---
    d_domain: float = 0.05          # rozmiar domeny [m] (5 cm)
    l_channel: float = 0.025        # długość kanału [m] (2.5 cm)
    r_inner: float = 0.035          # promień wewnętrzny [m]
    r_outer: float = 0.050          # promień zewnętrzny [m]

    # --- warunki wyładowania ---
    V_applied: float = 300.0        # napięcie [V] (STAŁE, nie obwód RLC)
    eps_anode_eV: float = 3.0       # średnia energia elektronów na anodzie
    eps_cathode_eV: float = 3.0     # średnia energia elektronów na katodzie

    # --- pole magnetyczne ---
    B_max: float = 0.015            # 150 G [T]
    sigma_B_in: float = 0.011       # szerokość profilu wewnątrz kanału [m]
    sigma_B_out: float = 0.018      # szerokość profilu na zewnątrz [m]

    # --- neutrale ---
    mdot_mg_s: float = 5.0          # przepływ ksenonu [mg/s]
    v_neutral: float = 150.0        # prędkość wstrzykiwanych atomów [m/s]
    neutral_influx: float = 5.7257e21   # strumień atomów na anodzie [m⁻²s⁻¹]
    m_ion: float = 131.293 * AMU    # masa Xe

    # --- transport anomalny (NARZUCONY przez LANDMARK) ---
    k_m: float = 2.5e-13            # stały współczynnik pędu e-n [m³/s]
    nu_w_in: float = 1.0e7          # anomalna częstość "ścienna" w kanale [s⁻¹]
    nu_w_out: float = 0.0           # poza kanałem
    alpha_in: float = 0.1           # współczynnik "Bohma" w kanale
    alpha_out: float = 1.0          # poza kanałem
    U_loss_eV: float = 20.0         # W = ν_ε·exp(−U/ε), U = 20 eV

    # --- przypadki testowe (różnią się TYLKO ν_ε w kanale) ---
    case: int = 1

    def __post_init__(self):
        # ν_ε: częstość strat energii na ściance
        nu_eps_in = {1: 1.0e7, 2: 0.5e7, 3: 0.4e7}
        if self.case not in nu_eps_in:
            raise ValueError("case musi być 1, 2 albo 3")
        self.nu_eps_in = nu_eps_in[self.case]
        self.nu_eps_out = 1.0e7

        # pole przekroju kanału
        self.A_channel = np.pi * (self.r_outer**2 - self.r_inner**2)
        # gęstość neutrali odpowiadająca strumieniowi wlotowemu
        self.n_neutral_inlet = self.neutral_influx / self.v_neutral

    # ---------------- profile przestrzenne ----------------
    def B_profile(self, x):
        """B(x) = B_max·exp(−(x−l)²/(2σ_B²)), σ_B różne wewnątrz i na zewnątrz."""
        x = np.asarray(x, dtype=float)
        sigma = np.where(x <= self.l_channel, self.sigma_B_in, self.sigma_B_out)
        return self.B_max * np.exp(-((x - self.l_channel) ** 2) / (2.0 * sigma ** 2))

    def nu_wall(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= self.l_channel, self.nu_w_in, self.nu_w_out)

    def alpha_bohm(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= self.l_channel, self.alpha_in, self.alpha_out)

    def omega_ce(self, x):
        return E_CHARGE * self.B_profile(x) / M_ELECTRON

    def nu_anomalous(self, x):
        """Anomalna częstość pędu narzucona przez LANDMARK: ν_w + α·ω_ce/16.

        To jest CAŁY mechanizm transportu poprzecznego w LANDMARK. Kod PIC 1D
        osiowy nie ma go z natury, więc bez dodania tego członu (jako izotropowo
        rozpraszającego procesu w MCC) żadne porównanie transportu nie ma sensu.
        """
        return self.nu_wall(x) + self.alpha_bohm(x) * self.omega_ce(x) / 16.0

    def nu_momentum(self, x, n_neutral):
        """Pełna częstość pędu LANDMARK: ν_m = N·k_m + ν_w + α·ω_ce/16."""
        return n_neutral * self.k_m + self.nu_anomalous(x)
