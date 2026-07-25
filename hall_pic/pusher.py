"""Operacje siatka<->cząstka: depozycja (CIC), zbieranie pola, pchacz Borisa,
warunki brzegowe (absorpcja na anodzie/katodzie) i emisja z katody.

Model 1D3V: położenie tylko x, ale prędkość 3-składnikowa (vx, vy, vz).
Pole elektryczne E = (Ex, 0, 0) wzdłuż osi; pole magnetyczne B = (0, By, 0)
radialne. Kombinacja E_x i B_y daje dryf ExB w kierunku azymutalnym (z) —
kluczowy dla fizyki silnika Halla i zamagnetyzowania elektronów.
"""

import numpy as np

from .constants import E_CHARGE


def deposit_charge(species_list, cfg):
    """Zwraca gęstość ładunku rho na węzłach [C/m^3] (CIC, 1. rząd)."""
    rho = np.zeros(cfg.n_nodes)
    dx = cfg.dx
    for sp in species_list:
        if sp.N == 0:
            continue
        xp = sp.ax
        fi = xp / dx
        i = np.floor(fi).astype(np.int64)
        i = np.clip(i, 0, cfg.Nx - 1)
        frac = fi - i
        contrib = sp.charge * sp.aw / dx   # [C/m^3] * (bez dx bo dzielimy)
        # rozkład na węzły i, i+1
        np.add.at(rho, i, contrib * (1.0 - frac))
        np.add.at(rho, i + 1, contrib * frac)
    # węzły brzegowe reprezentują pół komórki -> podwajamy gęstość na brzegu
    rho[0] *= 2.0
    rho[-1] *= 2.0
    return rho


def deposit_number_density(sp, cfg):
    """Gęstość liczbowa n(x) [1/m^3] pojedynczego gatunku (do diagnostyki)."""
    n = np.zeros(cfg.n_nodes)
    if sp.N == 0:
        return n
    dx = cfg.dx
    fi = sp.ax / dx
    i = np.clip(np.floor(fi).astype(np.int64), 0, cfg.Nx - 1)
    frac = fi - i
    contrib = sp.aw / dx
    np.add.at(n, i, contrib * (1.0 - frac))
    np.add.at(n, i + 1, contrib * frac)
    n[0] *= 2.0
    n[-1] *= 2.0
    return n


def gather_field(sp, E_nodes, cfg):
    """Interpoluje E_x z węzłów do położeń cząstek (CIC)."""
    dx = cfg.dx
    fi = sp.ax / dx
    i = np.clip(np.floor(fi).astype(np.int64), 0, cfg.Nx - 1)
    frac = fi - i
    return E_nodes[i] * (1.0 - frac) + E_nodes[i + 1] * frac


def boris_push(sp, Ex_p, cfg):
    """Pchacz Borisa 3V. Aktualizuje prędkość (E_x, B_y) i położenie x.

    B_y zależy od położenia cząstki (profil radialny). E działa tylko wzdłuż x.
    """
    if sp.N == 0:
        return
    dt = cfg.dt
    qm = sp.charge / sp.mass
    By = cfg.B_profile(sp.ax)          # indukcja w miejscu każdej cząstki

    # pół-przyspieszenie elektryczne (tylko x)
    vminus_x = sp.avx + qm * Ex_p * 0.5 * dt
    vminus_y = sp.avy.copy()
    vminus_z = sp.avz.copy()

    # obrót magnetyczny wokół osi y: t = (0, ty, 0)
    ty = qm * By * 0.5 * dt
    s_fac = 2.0 * ty / (1.0 + ty * ty)

    # v' = vminus + vminus x t   (t = (0,ty,0))
    # (a x b) dla b=(0,ty,0): (az*ty? ) -> policzmy jawnie:
    # v x t = (vy*tz - vz*ty, vz*tx - vx*tz, vx*ty - vy*tx), tx=tz=0
    #       = (-vz*ty, 0, vx*ty)
    vprime_x = vminus_x + (-vminus_z * ty)
    vprime_y = vminus_y
    vprime_z = vminus_z + (vminus_x * ty)

    # vplus = vminus + vprime x s,  s = (0, sy, 0)
    # vprime x s = (-vprime_z*sy, 0, vprime_x*sy)
    vplus_x = vminus_x + (-vprime_z * s_fac)
    vplus_y = vminus_y
    vplus_z = vminus_z + (vprime_x * s_fac)

    # druga połowa przyspieszenia elektrycznego
    sp.vx[:sp.N] = vplus_x + qm * Ex_p * 0.5 * dt
    sp.vy[:sp.N] = vplus_y
    sp.vz[:sp.N] = vplus_z

    # aktualizacja położenia (tylko x)
    sp.x[:sp.N] = sp.ax + sp.avx * dt


def apply_boundaries(sp, cfg):
    """Absorpcja na anodzie (x<0) i katodzie (x>L). Zwraca zebrany ładunek
    [C/m^2] na anodzie i katodzie (dodatni = ilość ładunku danego gatunku).

    Zwraca (q_anode, q_cathode, n_ion_to_cathode) gdzie ładunki są sumami
    q_particle * w po zaabsorbowanych cząstkach.
    """
    if sp.N == 0:
        return 0.0, 0.0, 0.0
    x = sp.ax
    hit_anode = x < 0.0
    hit_cath = x > cfg.L
    q_anode = sp.charge * np.sum(sp.aw[hit_anode]) if np.any(hit_anode) else 0.0
    q_cath = sp.charge * np.sum(sp.aw[hit_cath]) if np.any(hit_cath) else 0.0
    n_cath_count = float(np.sum(sp.aw[hit_cath])) if np.any(hit_cath) else 0.0
    kill = hit_anode | hit_cath
    sp.remove_mask(kill)
    return q_anode, q_cath, n_cath_count


def inject_cathode_electrons(electrons, w_ref, n_inject_weight, cfg, rng):
    """Wstrzykuje elektrony z płaszczyzny katody (x=L) do wnętrza.

    n_inject_weight — łączna waga [1/m^2] do wstrzyknięcia (np. proporcjonalna
    do strumienia jonów uciekających do katody, * cathode_gain).
    Rozkład prędkości: półmaxwellowski skierowany w -x (do wnętrza kanału).
    """
    if n_inject_weight <= 0.0:
        return
    n_new = int(round(n_inject_weight / w_ref))
    if n_new <= 0:
        return
    from .constants import M_ELECTRON, K_BOLTZMANN
    vth = np.sqrt(E_CHARGE * cfg.Te_cathode_eV / M_ELECTRON)
    # prędkości maxwellowskie, vx skierowane do wnętrza (ujemne)
    vx = -np.abs(rng.normal(0.0, vth, n_new))
    vy = rng.normal(0.0, vth, n_new)
    vz = rng.normal(0.0, vth, n_new)
    # tuż przy katodzie
    x0 = cfg.L - 1e-6 - rng.random(n_new) * (cfg.dx * 0.5)
    electrons.add(x0, vx, vy, vz, np.full(n_new, w_ref))
