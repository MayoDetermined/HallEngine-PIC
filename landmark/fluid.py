"""Model płynowy LANDMARK 1D-axial — implementacja 1:1 ze specyfikacji.

Równania (wprost z dokumentu "Model-1D-HALL Benchmark", odczytane z PDF):

  ciągłość:      ∂n/∂t + ∂(n v_e)/∂x = ∂n/∂t + ∂(n v_i)/∂x = n N k_iz(ε)
  neutrale:      ∂N/∂t + v_n ∂N/∂x = −N n k_iz(ε)
  pęd elektronów: n v_e = μ n ∂Φ/∂x − μ ∂(n ε)/∂x
  pęd jonów:     ∂v_i/∂t + v_i ∂v_i/∂x − η ∂²v_i/∂x² + N k_iz v_i = −(e/m_i) ∂Φ/∂x
  energia:       ∂(nε)/∂t + ∂/∂x[(5/3) n v_e ε − (10/9) μ n ε ∂ε/∂x]
                     = −n v_e ∂Φ/∂x − n N K(ε) − n W

  μ = (e/m_e)·ν/(ν² + (eB/m_e)²)
  ν = N k_m + ν_w + (β/16)(eB/m_e)
  B = B_max exp(−(x−l)²/(2δ_B²))
  η = δ_η √(2eε/(3m_i))
  W = ν_ε · ε · exp(−U/ε)

Warunki brzegowe:
  Φ(0) = V, Φ(d) = 0
  ε(0) = ε_an = 3 eV, ε(d) = ε_cath = 3 eV
  N(0) = Γ_n/v_n − v_i(0)·n(0)/v_n     (wlot gazu + rekombinacja jonów na anodzie)

DOMKNIĘCIE NA PRĄD (istota modelu quasineutralnego):
  Z obu równań ciągłości: ∂/∂x(n v_i − n v_e) = 0, więc gęstość prądu
      j = e(n v_i − n v_e)
  jest JEDNORODNA w x (zależy tylko od czasu). Podstawiając n v_e = n v_i − j/e
  do równania pędu elektronów:
      ∂Φ/∂x = G(x) − j·H(x),   G = [n v_i + μ ∂(nε)/∂x]/(μn),  H = 1/(e μ n)
  a warunek Φ(d) − Φ(0) = −V daje
      j = (V + ∫₀^d G dx) / ∫₀^d H dx
  Nie ma tu równania Poissona — i właśnie dlatego model jest tani.
"""

import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON
from .config_lm import LandmarkConfig
from . import rates


class LandmarkFluid:
    def __init__(self, cfg: LandmarkConfig, Nx=200, delta_eta=0.5e-3,
                 n_init=1.0e17, eps_init=4.0, cfl=0.2):
        self.cfg = cfg
        self.Nx = Nx
        self.x = np.linspace(0.0, cfg.d_domain, Nx + 1)
        self.dx = self.x[1] - self.x[0]
        self.delta_eta = delta_eta
        self.cfl = cfl

        # tabele LANDMARK
        e_t, k_t, K_t = rates.load_landmark_tables()
        self._eps_tab, self._kiz_tab, self._K_tab = e_t, k_t, K_t

        # profile stałe w czasie
        self.B = cfg.B_profile(self.x)
        self.omega_ce = E_CHARGE * self.B / M_ELECTRON
        self.nu_w = cfg.nu_wall(self.x)
        self.beta = cfg.alpha_bohm(self.x)
        self.nu_eps = np.where(self.x <= cfg.l_channel, cfg.nu_eps_in, cfg.nu_eps_out)

        # stan początkowy
        self.n = np.full(Nx + 1, n_init)
        self.v_i = np.zeros(Nx + 1)
        self.N = np.full(Nx + 1, cfg.n_neutral_inlet)
        self.eps = np.full(Nx + 1, eps_init)
        self.eps[0] = cfg.eps_anode_eV
        self.eps[-1] = cfg.eps_cathode_eV

        self.t = 0.0
        self.j = 0.0
        self.Phi = np.zeros(Nx + 1)

        # Podłoga gęstości: w warunkach LANDMARK n w domenie nie schodzi poniżej
        # ~1e16 (szczyt ~1e18), a zbyt niska podłoga rozdmuchuje v_e = Γ_e/n
        # i niszczy krok czasowy.
        self.n_floor = 1.0e15
        self.eps_floor = 0.2
        self.eps_ceil = 150.0        # zakres tabel LANDMARK

    # ---------------- współczynniki ----------------
    def k_iz(self, eps):
        return np.interp(eps, self._eps_tab, self._kiz_tab)

    def K_loss(self, eps):
        return np.interp(eps, self._eps_tab, self._K_tab)

    def mobility(self):
        nu = self.N * self.cfg.k_m + self.nu_w + (self.beta / 16.0) * self.omega_ce
        return (E_CHARGE / M_ELECTRON) * nu / (nu**2 + self.omega_ce**2)

    # ---------------- domknięcie na prąd + potencjał ----------------
    def solve_current_and_potential(self, mu):
        n = np.maximum(self.n, self.n_floor)
        d_neps = np.gradient(n * self.eps, self.dx)
        G = (n * self.v_i + mu * d_neps) / (mu * n)
        H = 1.0 / (E_CHARGE * mu * n)
        intG = np.trapezoid(G, self.x)
        intH = np.trapezoid(H, self.x)
        j = (self.cfg.V_applied + intG) / intH        # [A/m²]
        dPhi = G - j * H
        # Φ(x) = V + ∫₀^x dΦ/dx'
        Phi = np.concatenate(([0.0], np.cumsum(0.5 * (dPhi[1:] + dPhi[:-1]) * self.dx)))
        Phi = self.cfg.V_applied + Phi
        return j, dPhi, Phi

    # ---------------- schematy różnicowe ----------------
    @staticmethod
    def _upwind_flux(q, u_face):
        """Strumień na ścianach metodą pod prąd; q w węzłach, u_face na ścianach."""
        return np.where(u_face >= 0.0, u_face * q[:-1], u_face * q[1:])

    def _div_flux(self, F_face):
        """Dywergencja strumienia ścianowego -> w węzłach wewnętrznych."""
        d = np.zeros_like(self.n)
        d[1:-1] = (F_face[1:] - F_face[:-1]) / self.dx
        return d

    def _implicit_energy_diffusion(self, neps_star, Dface, dt):
        """Niejawny (wstecz Eulera) krok dyfuzji energii elektronów.

        Rozwiązuje  n·ε − dt·∂/∂x[D ∂ε/∂x] = (nε)*  z Dirichletem na obu końcach.
        Układ trójdiagonalny; solve_banded jest w C, więc koszt jest znikomy,
        a znika ograniczenie dt ~ dx²/D (najostrzejsze w tym modelu).
        """
        from scipy.linalg import solve_banded
        cfg = self.cfg
        m = self.Nx + 1
        dx2 = self.dx * self.dx
        n = np.maximum(self.n, self.n_floor)

        lo = np.zeros(m)      # poddiagonala
        di = np.zeros(m)      # diagonala
        up = np.zeros(m)      # naddiagonala
        rhs = neps_star.copy()

        aW = dt * Dface[:-1] / dx2      # sprzężenie z węzłem i-1 (ściany 0..m-3)
        aE = dt * Dface[1:] / dx2       # sprzężenie z węzłem i+1
        di[1:-1] = n[1:-1] + aW + aE
        lo[1:-1] = -aW
        up[1:-1] = -aE

        # brzegi: Dirichlet ε = ε_an / ε_cath
        di[0] = 1.0;  up[0] = 0.0;  rhs[0] = cfg.eps_anode_eV
        di[-1] = 1.0; lo[-1] = 0.0; rhs[-1] = cfg.eps_cathode_eV

        # ab w formacie solve_banded: wiersz0=naddiag (przesunięty), 1=diag, 2=poddiag
        ab = np.zeros((3, m))
        ab[0, 1:] = up[:-1]
        ab[1, :] = di
        ab[2, :-1] = lo[1:]
        eps_new = solve_banded((1, 1), ab, rhs)
        return np.clip(eps_new, self.eps_floor, self.eps_ceil)

    # ---------------- krok czasowy ----------------
    def step(self):
        cfg = self.cfg
        dx = self.dx
        n = np.maximum(self.n, self.n_floor)
        eps = np.clip(self.eps, self.eps_floor, self.eps_ceil)

        mu = self.mobility()
        j, dPhi, Phi = self.solve_current_and_potential(mu)
        self.j, self.Phi = j, Phi

        kiz = self.k_iz(eps)
        K = self.K_loss(eps)
        W = self.nu_eps * eps * np.exp(-cfg.U_loss_eV / eps)
        eta = self.delta_eta * np.sqrt(2.0 * E_CHARGE * eps / (3.0 * cfg.m_ion))

        # STRUMIEŃ elektronów z jednorodności prądu. Używamy Γ_e, nie v_e:
        # v_e = v_i − j/(e·n) rozbiega się przy małym n (przy anodzie), podczas
        # gdy sam strumień Γ_e = n·v_i − j/e pozostaje ograniczony. Liczenie
        # członów energii przez Γ_e zamiast v_e usuwa tę osobliwość.
        Gamma_e = n * self.v_i - j / E_CHARGE
        v_e = Gamma_e / n

        # --- dobór kroku czasowego ---
        # Dyfuzja energii elektronów jest liczona NIEJAWNIE (patrz niżej), więc
        # nie ogranicza kroku; zostaje adwekcja i lepkość jonowa. To zdejmuje
        # najostrzejsze ograniczenie (~dx²/D) i daje ~8x większy krok.
        v_max = max(np.abs(self.v_i).max(), np.abs(v_e).max(), cfg.v_neutral, 1.0)
        D_max = max(eta.max(), 1e-6)
        dt = self.cfl * min(dx / v_max, 0.5 * dx * dx / D_max)
        dt = min(dt, 1.0e-9)

        # --- ciągłość jonów ---
        vi_face = 0.5 * (self.v_i[:-1] + self.v_i[1:])
        Fn = self._upwind_flux(n, vi_face)
        dn = -self._div_flux(Fn) + n * self.N * kiz

        # --- neutrale (v_n > 0, pod prąd wstecz) ---
        dN = np.zeros_like(self.N)
        dN[1:-1] = -cfg.v_neutral * (self.N[1:-1] - self.N[:-2]) / dx \
                   - self.N[1:-1] * n[1:-1] * kiz[1:-1]
        dN[-1] = -cfg.v_neutral * (self.N[-1] - self.N[-2]) / dx \
                 - self.N[-1] * n[-1] * kiz[-1]

        # --- pęd jonów ---
        dvi = np.zeros_like(self.v_i)
        adv = np.where(self.v_i[1:-1] >= 0,
                       (self.v_i[1:-1] - self.v_i[:-2]) / dx,
                       (self.v_i[2:] - self.v_i[1:-1]) / dx)
        lap = (self.v_i[2:] - 2.0 * self.v_i[1:-1] + self.v_i[:-2]) / dx**2
        dvi[1:-1] = (-self.v_i[1:-1] * adv
                     + eta[1:-1] * lap
                     - self.N[1:-1] * kiz[1:-1] * self.v_i[1:-1]
                     - (E_CHARGE / cfg.m_ion) * dPhi[1:-1])

        # --- energia elektronów ---
        neps = n * eps
        # konwekcja (5/3)·Γ_e·ε — strumień na ścianach z Γ_e (ograniczone),
        # ε brane pod prąd względem znaku Γ_e
        Ge_face = 0.5 * (Gamma_e[:-1] + Gamma_e[1:])
        eps_up = np.where(Ge_face >= 0.0, eps[:-1], eps[1:])
        Fc = (5.0 / 3.0) * Ge_face * eps_up
        # dyfuzja (10/9) μ n ε ∂ε/∂x — liczona NIEJAWNIE po kroku jawnym
        Dface = (10.0 / 9.0) * 0.5 * ((mu * n * eps)[:-1] + (mu * n * eps)[1:])
        dneps = -self._div_flux(Fc)
        # Człon Joule'a: ZNAK POPRAWIONY względem PDF (patrz uwaga niżej).
        # Standardowa postać to −e·Γ_e·E; ponieważ E = −∂Φ/∂x, daje to
        # +n·v_e·∂Φ/∂x. W dokumencie LANDMARK zapisano "−n v_e ∂Φ/∂x", co przy
        # v_e<0 i ∂Φ/∂x<0 CHŁODZI elektrony i nie pozwala odtworzyć ε≈42 eV
        # z Figure 2 — w PDF pomylono E z ∂Φ/∂x.
        dneps[1:-1] += (+Gamma_e[1:-1] * dPhi[1:-1]
                        - n[1:-1] * self.N[1:-1] * K[1:-1]
                        - n[1:-1] * W[1:-1])

        # --- aktualizacja ---
        self.n = np.maximum(self.n + dt * dn, self.n_floor)
        self.N = np.maximum(self.N + dt * dN, 0.0)
        self.v_i = self.v_i + dt * dvi
        neps_star = np.maximum(neps + dt * dneps, self.n_floor * self.eps_floor)
        # krok niejawny dla dyfuzji energii: n·ε^{k+1} − dt·∂/∂x[D ∂ε^{k+1}/∂x] = (nε)*
        self.eps = self._implicit_energy_diffusion(neps_star, Dface, dt)

        # --- warunki brzegowe ---
        self.n[0], self.n[-1] = self.n[1], self.n[-2]           # wypływ
        self.v_i[0], self.v_i[-1] = self.v_i[1], self.v_i[-2]
        self.eps[0] = cfg.eps_anode_eV
        self.eps[-1] = cfg.eps_cathode_eV
        # neutrale na anodzie: wlot gazu + rekombinacja jonów
        self.N[0] = max(cfg.neutral_influx / cfg.v_neutral
                        - self.v_i[0] * self.n[0] / cfg.v_neutral, 0.0)

        self.t += dt
        return dt

    # ---------------- wielkości pochodne ----------------
    @property
    def discharge_current(self):
        """Prąd wyładowania [A] = j · pole przekroju kanału."""
        return self.j * self.cfg.A_channel

    def ionization_rate(self):
        return self.n * self.N * self.k_iz(np.clip(self.eps, self.eps_floor, self.eps_ceil))

    def E_field(self):
        return -np.gradient(self.Phi, self.dx)
