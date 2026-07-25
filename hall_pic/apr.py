"""Adaptive Particle Refinement (APR) — lepsza statystyka wiązek RE.

Strategia:
  * DETEKCJA WIĄZKI: w każdej komórce liczymy frakcję elektronów o energii
    > E_RE (elektrony "runaway"). Komórki, gdzie frakcja przekracza próg,
    są flagowane jako "wiązkowe" — tam adaptacyjnie ZAGĘSZCZAMY (split).
  * SPLIT (zagęszczanie): elektron w komórce wiązkowej dzielony na 2 cząstki
    o połowie wagi, z minimalnym rozrzutem położenia w obrębie komórki i tą
    samą prędkością. Zachowuje masę, pęd i energię (moment 0/1/2), a zwiększa
    liczbę próbek -> gładsza funkcja rozkładu w ogonie RE.
  * MERGE (rozrzedzanie): w komórkach NIE-wiązkowych z nadmiarem cząstek łączymy
    pary o zbliżonej prędkości w jedną (waga sumaryczna, prędkość ważona masą).
    Zachowuje masę i pęd; niewielka strata energii akceptowalna poza wiązką,
    gdzie i tak nie badamy ogona.

Waga referencyjna w_ref definiuje "pełną" superczątstkę; split nie schodzi
poniżej apr_min_weight_ratio * w_ref, by nie eksplodować liczbą cząstek.
"""

import numpy as np

from .constants import E_CHARGE


def _cell_index(x, cfg):
    return np.clip(np.floor(x / cfg.dx).astype(np.int64), 0, cfg.Nx - 1)


def flag_beam_cells(electrons, cfg):
    """Zwraca (flaga_wiazki[Nx] bool, frakcja_RE[Nx])."""
    frac = np.zeros(cfg.Nx)
    counts = np.zeros(cfg.Nx)
    if electrons.N == 0:
        return frac.astype(bool), frac
    ci = _cell_index(electrons.ax, cfg)
    eps = electrons.kinetic_energy_eV()
    w = electrons.aw
    tot_w = np.zeros(cfg.Nx)
    re_w = np.zeros(cfg.Nx)
    np.add.at(tot_w, ci, w)
    np.add.at(re_w, ci, w * (eps > cfg.E_RE_eV))
    np.add.at(counts, ci, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(tot_w > 0, re_w / tot_w, 0.0)
    flags = (frac >= cfg.apr_beam_frac_threshold) & (counts > 0)
    return flags, frac


def refine_split(electrons, cfg, w_ref, rng):
    """Dzieli elektrony w komórkach wiązkowych, dążąc do apr_split_target_ppc."""
    if electrons.N == 0:
        return 0
    flags, _ = flag_beam_cells(electrons, cfg)
    if not np.any(flags):
        return 0
    ci = _cell_index(electrons.ax, cfg)
    counts = np.bincount(ci, minlength=cfg.Nx)

    w_min = cfg.apr_min_weight_ratio * w_ref
    n_split_total = 0
    new_x = []; new_vx = []; new_vy = []; new_vz = []; new_w = []

    for c in np.nonzero(flags)[0]:
        if counts[c] >= cfg.apr_split_target_ppc:
            continue
        in_cell = np.nonzero(ci == c)[0]
        if in_cell.size == 0:
            continue
        # ile jeszcze cząstek chcemy dodać w tej komórce
        deficit = cfg.apr_split_target_ppc - counts[c]
        # dziel najcięższe cząstki (największy wkład statystyczny)
        wsel = electrons.w[in_cell]
        order = in_cell[np.argsort(-wsel)]
        n_to_split = min(deficit, order.size)
        chosen = order[:n_to_split]
        # tylko te, których waga pozwala na podział
        chosen = chosen[electrons.w[chosen] > 2.0 * w_min]
        if chosen.size == 0:
            continue
        half = electrons.w[chosen] * 0.5
        electrons.w[chosen] = half           # rodzic dostaje połowę wagi
        # dziecko: ta sama prędkość, minimalny rozrzut położenia w komórce
        jitter = (rng.random(chosen.size) - 0.5) * cfg.dx * 0.1
        cx = np.clip(electrons.x[chosen] + jitter, 1e-9, cfg.L - 1e-9)
        new_x.append(cx)
        new_vx.append(electrons.vx[chosen].copy())
        new_vy.append(electrons.vy[chosen].copy())
        new_vz.append(electrons.vz[chosen].copy())
        new_w.append(half.copy())
        n_split_total += chosen.size

    if n_split_total > 0:
        electrons.add(np.concatenate(new_x), np.concatenate(new_vx),
                      np.concatenate(new_vy), np.concatenate(new_vz),
                      np.concatenate(new_w))
    return n_split_total


def coarsen_merge(electrons, cfg, rng):
    """Łączy pary elektronów w komórkach NIE-wiązkowych z nadmiarem cząstek."""
    if electrons.N == 0:
        return 0
    flags, _ = flag_beam_cells(electrons, cfg)
    ci = _cell_index(electrons.ax, cfg)
    counts = np.bincount(ci, minlength=cfg.Nx)

    kill = np.zeros(electrons.N, dtype=bool)
    n_merged = 0

    over = np.nonzero((counts > cfg.apr_max_ppc) & (~flags))[0]
    for c in over:
        in_cell = np.nonzero(ci == c)[0]
        n_excess = in_cell.size - cfg.apr_max_ppc
        if n_excess <= 1:
            continue
        # sortuj po v_x aby łączyć podobne prędkości (mały błąd energii)
        vx = electrons.vx[in_cell]
        order = in_cell[np.argsort(vx)]
        # łącz kolejne pary aż zredukujemy nadmiar
        n_pairs = n_excess  # każda para usuwa 1 cząstkę
        k = 0
        for p in range(0, order.size - 1, 2):
            if k >= n_pairs:
                break
            a, b = order[p], order[p + 1]
            wa, wb = electrons.w[a], electrons.w[b]
            wsum = wa + wb
            if wsum <= 0:
                continue
            # prędkość ważona wagą (zachowuje pęd i masę)
            electrons.vx[a] = (wa*electrons.vx[a] + wb*electrons.vx[b]) / wsum
            electrons.vy[a] = (wa*electrons.vy[a] + wb*electrons.vy[b]) / wsum
            electrons.vz[a] = (wa*electrons.vz[a] + wb*electrons.vz[b]) / wsum
            electrons.x[a] = (wa*electrons.x[a] + wb*electrons.x[b]) / wsum
            electrons.w[a] = wsum
            kill[b] = True
            k += 1
            n_merged += 1

    if n_merged > 0:
        electrons.remove_mask(kill)
    return n_merged


def run_apr(electrons, cfg, w_ref, rng):
    """Pełny cykl APR: najpierw rozrzedzanie, potem zagęszczanie wiązki."""
    n_merged = coarsen_merge(electrons, cfg, rng)
    n_split = refine_split(electrons, cfg, w_ref, rng)
    return n_split, n_merged
