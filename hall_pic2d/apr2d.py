"""Adaptive Particle Refinement w 2D — w pełni zwektoryzowany.

Ta sama strategia co w 1D (split w komórkach wiązkowych, merge w tle), ale
przy N1·N2 komórkach pętla pythonowa byłaby wąskim gardłem. Zamiast niej:

    order  = lexsort((klucz_wtórny, cell))   -> cząstki pogrupowane po komórkach
    starts = searchsorted(cell_posortowane)  -> początek każdej komórki
    rank   = arange(N) − starts[cell]        -> ranga cząstki WEWNĄTRZ komórki

Mając rangę, wybór "pierwszych k cząstek w każdej komórce" to jedno porównanie
wektorowe `rank < k[cell]`. Split preferuje najcięższe cząstki (klucz −w),
merge sortuje po v1, żeby łączyć cząstki o zbliżonej prędkości.
"""

import numpy as np

from hall_pic.collisions import isotropic_unit_vectors


def _cell_idx(sp, cfg):
    i = np.clip(np.floor(sp.ax1 / cfg.h1).astype(np.int64), 0, cfg.N1 - 1)
    j = np.floor(sp.ax2 / cfg.h2).astype(np.int64)
    if cfg.x2_bc == "periodic":
        j = np.mod(j, cfg.N2)
    else:
        np.clip(j, 0, cfg.N2 - 1, out=j)
    return i * cfg.N2 + j


def flag_beam_cells(sp, cfg, cell=None):
    """Komórka jest 'wiązkowa', gdy frakcja wagi elektronów o E > E_RE
    przekracza próg. Zwraca (flagi[n_cells] bool, frakcja[n_cells])."""
    if sp.N == 0:
        z = np.zeros(cfg.n_cells)
        return z.astype(bool), z
    if cell is None:
        cell = _cell_idx(sp, cfg)
    eps = sp.kinetic_energy_eV()
    tot = np.bincount(cell, weights=sp.aw, minlength=cfg.n_cells)
    re = np.bincount(cell, weights=sp.aw * (eps > cfg.E_RE_eV), minlength=cfg.n_cells)
    frac = np.where(tot > 0, re / np.maximum(tot, 1e-300), 0.0)
    return frac >= cfg.apr_beam_frac_threshold, frac


def _ranks_in_cells(cell, order, cfg):
    """Ranga każdej cząstki wewnątrz jej komórki (w kolejności `order`)."""
    sc = cell[order]
    starts = np.searchsorted(sc, np.arange(cfg.n_cells), side="left")
    rank = np.arange(sc.size) - starts[sc]
    return sc, rank


def refine_split(electrons, cfg, w_ref, rng):
    """Dzieli najcięższe elektrony w komórkach wiązkowych do apr_split_target_ppc."""
    if electrons.N == 0 or electrons.N >= cfg.max_particles:
        return 0                     # twardy limit — chroni przed lawiną podziałów
    cell = _cell_idx(electrons, cfg)
    flags, _ = flag_beam_cells(electrons, cfg, cell)
    if not np.any(flags):
        return 0
    counts = np.bincount(cell, minlength=cfg.n_cells)
    deficit = np.where(flags, np.maximum(cfg.apr_split_target_ppc - counts, 0), 0)
    if deficit.sum() == 0:
        return 0

    w_min = cfg.apr_min_weight_ratio * w_ref
    # grupuj po komórce; wewnątrz komórki najcięższe najpierw
    order = np.lexsort((-electrons.aw, cell))
    sc, rank = _ranks_in_cells(cell, order, cfg)
    take = (rank < deficit[sc]) & (electrons.aw[order] > 2.0 * w_min)
    chosen = order[take]
    if chosen.size == 0:
        return 0

    # odczyt PRZED add() — add może realokować tablice
    half = electrons.w[chosen] * 0.5
    v1 = electrons.v1[chosen].copy()
    v2 = electrons.v2[chosen].copy()
    v3 = electrons.v3[chosen].copy()
    x1 = electrons.x1[chosen] + (rng.random(chosen.size) - 0.5) * cfg.h1 * 0.1
    x2 = electrons.x2[chosen] + (rng.random(chosen.size) - 0.5) * cfg.h2 * 0.1
    x1 = np.clip(x1, 1e-9, cfg.L1 - 1e-9)
    if cfg.x2_bc == "periodic":
        x2 = np.mod(x2, cfg.L2)
    else:
        x2 = np.clip(x2, 1e-9, cfg.L2 - 1e-9)

    electrons.w[chosen] = half              # rodzic zachowuje połowę wagi
    electrons.add(x1, x2, v1, v2, v3, half.copy())
    return int(chosen.size)


def coarsen_merge(electrons, cfg, rng):
    """Łączy grupy 4 elektronów w 2, ZACHOWUJĄC masę, pęd I ENERGIĘ.

    Schemat typu Vranica. Dla grupy o łącznej wadze W, prędkości średniej u
    i średnim kwadracie prędkości E = Σw|v|²/W, dwie cząstki wynikowe mają
    wagę W/2 i prędkości u ± d, gdzie |d|² = E − |u|², a kierunek d jest
    losowy izotropowy. Wtedy:
        masa   : 2·(W/2) = W                                   ✔
        pęd    : (W/2)(u+d) + (W/2)(u−d) = W·u                 ✔
        energia: (W/2)(|u+d|² + |u−d|²) = W(|u|² + |d|²) = W·E  ✔

    Naiwne łączenie 2→1 przez uśrednianie prędkości zachowuje masę i pęd,
    ale gubi energię (sztuczne chłodzenie tła — w teście ~33%), dlatego
    nie jest używane.

    Grupa 4→2 usuwa 2 cząstki, czyli tyle samo na uczestnika co para 2→1.
    """
    if electrons.N == 0:
        return 0
    cell = _cell_idx(electrons, cfg)
    flags, _ = flag_beam_cells(electrons, cfg, cell)
    counts = np.bincount(cell, minlength=cfg.n_cells)
    over = (~flags) & (counts > cfg.apr_max_ppc)
    excess = np.where(over, counts - cfg.apr_max_ppc, 0)
    # każda grupa (4 uczestników) usuwa 2 cząstki
    n_groups = np.where(over, np.minimum(excess // 2, counts // 4), 0)
    total_groups = int(n_groups.sum())
    if total_groups == 0:
        return 0

    order = np.lexsort((electrons.av1, cell))   # w komórce po v1 (podobne prędkości)
    sc, rank = _ranks_in_cells(cell, order, cfg)
    part = rank < 4 * n_groups[sc]
    if not np.any(part):
        return 0

    idx = order[part]
    # globalny identyfikator grupy = przesunięcie komórki + numer grupy w komórce
    goff = np.concatenate(([0], np.cumsum(n_groups)[:-1]))
    gid = goff[sc[part]] + (rank[part] // 4)
    slot = rank[part] % 4

    w = electrons.w[idx]
    v1 = electrons.v1[idx]; v2 = electrons.v2[idx]; v3 = electrons.v3[idx]
    W = np.bincount(gid, weights=w, minlength=total_groups)
    P1 = np.bincount(gid, weights=w*v1, minlength=total_groups)
    P2 = np.bincount(gid, weights=w*v2, minlength=total_groups)
    P3 = np.bincount(gid, weights=w*v3, minlength=total_groups)
    Es = np.bincount(gid, weights=w*(v1*v1 + v2*v2 + v3*v3), minlength=total_groups)

    Wsafe = np.maximum(W, 1e-300)
    u1, u2, u3 = P1/Wsafe, P2/Wsafe, P3/Wsafe
    d2 = np.maximum(Es/Wsafe - (u1*u1 + u2*u2 + u3*u3), 0.0)
    d = np.sqrt(d2)
    e1, e2, e3 = isotropic_unit_vectors(total_groups, rng)

    # sloty 0 i 1 przeżywają jako para (u+d, u−d); sloty 2 i 3 znikają
    for s, sign in ((0, +1.0), (1, -1.0)):
        m = slot == s
        ii = idx[m]
        g = gid[m]
        electrons.w[ii] = 0.5 * W[g]
        electrons.v1[ii] = u1[g] + sign * d[g] * e1[g]
        electrons.v2[ii] = u2[g] + sign * d[g] * e2[g]
        electrons.v3[ii] = u3[g] + sign * d[g] * e3[g]
    # położenia przeżywających zostają bez zmian (uśrednianie psułoby szew periodyczny)

    kill = np.zeros(electrons.N, dtype=bool)
    kill[idx[slot >= 2]] = True
    electrons.remove_mask(kill)
    return 2 * total_groups


def run_apr(electrons, cfg, w_ref, rng):
    n_merged = coarsen_merge(electrons, cfg, rng)
    n_split = refine_split(electrons, cfg, w_ref, rng)
    return n_split, n_merged
