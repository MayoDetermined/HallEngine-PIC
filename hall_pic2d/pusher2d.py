"""Operacje siatka<->cząstka w 2D: depozycja dwuliniowa (CIC), zbieranie pola,
ogólny pchacz Borisa (dowolny 3-wektor B) i warunki brzegowe obu geometrii.

Depozycja używa np.bincount na spłaszczonych indeksach — istotnie szybciej niż
np.add.at przy setkach tysięcy cząstek (kluczowe, bo działamy w czystym NumPy).
"""

import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON


def cic_weights(sp, cfg):
    """Indeksy 4 węzłów i wagi dwuliniowe dla każdej cząstki.

    Wynik można ZBUFOROWAĆ w obrębie kroku: depozycja i zbieranie pola
    zachodzą przed pchaczem, czyli przy identycznych położeniach cząstek.
    """
    f1 = sp.ax1 / cfg.h1
    i = np.floor(f1).astype(np.int64)
    np.clip(i, 0, cfg.N1 - 1, out=i)
    a = f1 - i

    f2 = sp.ax2 / cfg.h2
    j = np.floor(f2).astype(np.int64)
    b = f2 - j
    if cfg.x2_bc == "periodic":
        j = np.mod(j, cfg.N2)
        j2 = np.mod(j + 1, cfg.N2)
    else:
        np.clip(j, 0, cfg.N2 - 1, out=j)
        j2 = j + 1

    n2n = cfg.n2_nodes
    # spłaszczone indeksy 4 narożników
    idx00 = i * n2n + j
    idx10 = (i + 1) * n2n + j
    idx01 = i * n2n + j2
    idx11 = (i + 1) * n2n + j2
    w00 = (1.0 - a) * (1.0 - b)
    w10 = a * (1.0 - b)
    w01 = (1.0 - a) * b
    w11 = a * b
    return (idx00, idx10, idx01, idx11), (w00, w10, w01, w11)


def _accumulate(idxs, wts, values, n_total):
    out = np.zeros(n_total)
    for idx, wt in zip(idxs, wts):
        out += np.bincount(idx, weights=values * wt, minlength=n_total)
    return out


def deposit_charge(species_list, cfg, cached=None):
    """Gęstość ładunku ρ na węzłach [C/m³], kształt (n1_nodes, n2_nodes).

    `cached` — opcjonalna lista (idxs, wts) z cic_weights() dla każdego gatunku.
    """
    n_total = cfg.n1_nodes * cfg.n2_nodes
    rho_flat = np.zeros(n_total)
    cell_vol = cfg.h1 * cfg.h2          # [m²]; waga w [1/m] -> n [1/m³]
    for k, sp in enumerate(species_list):
        if sp.N == 0:
            continue
        idxs, wts = cached[k] if cached is not None else cic_weights(sp, cfg)
        vals = sp.charge * sp.aw / cell_vol
        rho_flat += _accumulate(idxs, wts, vals, n_total)
    rho = rho_flat.reshape(cfg.n1_nodes, cfg.n2_nodes)
    _fix_boundary_nodes(rho, cfg)
    return rho


def deposit_number_density(sp, cfg):
    """Gęstość liczbowa n(x1,x2) [1/m³] jednego gatunku (diagnostyka)."""
    n_total = cfg.n1_nodes * cfg.n2_nodes
    if sp.N == 0:
        return np.zeros((cfg.n1_nodes, cfg.n2_nodes))
    idxs, wts = cic_weights(sp, cfg)
    vals = sp.aw / (cfg.h1 * cfg.h2)
    n = _accumulate(idxs, wts, vals, n_total).reshape(cfg.n1_nodes, cfg.n2_nodes)
    _fix_boundary_nodes(n, cfg)
    return n


def _fix_boundary_nodes(arr, cfg):
    """Węzły brzegowe obejmują pół komórki -> korekta objętości."""
    arr[0, :] *= 2.0
    arr[-1, :] *= 2.0
    if cfg.x2_bc != "periodic":
        arr[:, 0] *= 2.0
        arr[:, -1] *= 2.0


def gather_field(sp, F1, F2, cfg, cached=None):
    """Interpoluje (E1, E2) z węzłów do cząstek (dwuliniowo)."""
    idxs, wts = cached if cached is not None else cic_weights(sp, cfg)
    f1 = F1.ravel()
    f2 = F2.ravel()
    e1 = np.zeros(sp.N)
    e2 = np.zeros(sp.N)
    for idx, wt in zip(idxs, wts):
        e1 += f1[idx] * wt
        e2 += f2[idx] * wt
    return e1, e2


def boris_push(sp, E1p, E2p, cfg):
    """Pchacz Borisa 3V z DOWOLNYM 3-wektorem B (zależnym od geometrii).

    E leży w płaszczyźnie: E = (E1, E2, 0). B = B_vector(z).
    """
    if sp.N == 0:
        return
    dt = cfg.dt
    qm = sp.charge / sp.mass
    b1, b2, b3 = cfg.B_vector(sp.ax1)

    half = 0.5 * dt * qm
    # pół-przyspieszenie elektryczne
    vm1 = sp.av1 + half * E1p
    vm2 = sp.av2 + half * E2p
    vm3 = sp.av3.copy()

    # wektor obrotu t = qm·B·dt/2
    t1 = half * b1
    t2 = half * b2
    t3 = half * b3
    tsq = t1*t1 + t2*t2 + t3*t3
    s = 2.0 / (1.0 + tsq)
    s1, s2, s3 = s*t1, s*t2, s*t3

    # v' = v- + v- × t
    p1 = vm1 + (vm2*t3 - vm3*t2)
    p2 = vm2 + (vm3*t1 - vm1*t3)
    p3 = vm3 + (vm1*t2 - vm2*t1)

    # v+ = v- + v' × s
    vp1 = vm1 + (p2*s3 - p3*s2)
    vp2 = vm2 + (p3*s1 - p1*s3)
    vp3 = vm3 + (p1*s2 - p2*s1)

    # druga połowa przyspieszenia elektrycznego
    sp.v1[:sp.N] = vp1 + half * E1p
    sp.v2[:sp.N] = vp2 + half * E2p
    sp.v3[:sp.N] = vp3

    # ruch w płaszczyźnie
    sp.x1[:sp.N] = sp.ax1 + sp.av1 * dt
    sp.x2[:sp.N] = sp.ax2 + sp.av2 * dt


def apply_boundaries(sp, cfg):
    """Warunki brzegowe zależne od geometrii.

    x1: absorpcja na anodzie (z<0) i katodzie (z>L1) — w obu geometriach.
    x2: 'z-theta' -> zawijanie periodyczne; 'z-r' -> absorpcja na ściankach.

    Zwraca (Σw·q na anodzie, Σw·q na katodzie, Σw na katodzie, Σw na ściankach).
    """
    if sp.N == 0:
        return 0.0, 0.0, 0.0, 0.0

    # --- kierunek x2 ---
    wall_w = 0.0
    if cfg.x2_bc == "periodic":
        sp.x2[:sp.N] = np.mod(sp.ax2, cfg.L2)
        wall_kill = np.zeros(sp.N, dtype=bool)
    else:
        wall_kill = (sp.ax2 < 0.0) | (sp.ax2 > cfg.L2)
        if np.any(wall_kill):
            wall_w = float(np.sum(sp.aw[wall_kill]))

    # --- kierunek x1 ---
    hit_anode = sp.ax1 < 0.0
    hit_cath = sp.ax1 > cfg.L1
    q_anode = sp.charge * float(np.sum(sp.aw[hit_anode])) if np.any(hit_anode) else 0.0
    q_cath = sp.charge * float(np.sum(sp.aw[hit_cath])) if np.any(hit_cath) else 0.0
    w_cath = float(np.sum(sp.aw[hit_cath])) if np.any(hit_cath) else 0.0

    sp.remove_mask(hit_anode | hit_cath | wall_kill)
    return q_anode, q_cath, w_cath, wall_w


def inject_cathode_electrons(electrons, w_ref, inject_weight, cfg, rng):
    """Emisja elektronów z płaszczyzny katody (z = L1), półmaxwellowska do wnętrza."""
    if inject_weight <= 0.0:
        return
    n_new = int(round(inject_weight / w_ref))
    if n_new <= 0:
        return
    vth = np.sqrt(E_CHARGE * cfg.Te_cathode_eV / M_ELECTRON)
    v1 = -np.abs(rng.normal(0.0, vth, n_new))     # do wnętrza kanału
    v2 = rng.normal(0.0, vth, n_new)
    v3 = rng.normal(0.0, vth, n_new)
    x1 = cfg.L1 - 1e-6 - rng.random(n_new) * (cfg.h1 * 0.5)
    x2 = rng.random(n_new) * cfg.L2               # równomiernie wzdłuż x2
    electrons.add(x1, x2, v1, v2, v3, np.full(n_new, w_ref))
