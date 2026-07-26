"""Adaptacyjne dzielenie i łączenie czastek dla lepszej statystyki wiązki.

Wiązka rozpędzonych elektronów to niewielka mniejszość, więc łatwo o nią
statystycznie zubożeć. Dlatego tam, gdzie taka wiązka się tworzy, dzielimy
czastki na drobniejsze, żeby było ich więcej i żeby obraz był gładszy. Z kolei
w spokojnym tle, gdzie czastek robi się za dużo, łączymy je z powrotem, żeby
obliczenia nie spowalniały. Dzielenie zachowuje masę, pęd i energię, więc nie
zaburza fizyki, a jedynie poprawia dokładność. Łączenia nie schodzimy poniżej
pewnej najmniejszej wagi, żeby liczba czastek nie rosła bez opamiętania.
"""

import numpy as np

from .constants import E_CHARGE


def _cell_index(x, cfg):
    """Zwraca numer komórki, w której leży każda czastka."""
    return np.clip(np.floor(x / cfg.dx).astype(np.int64), 0, cfg.Nx - 1)


def flag_beam_cells(electrons, cfg):
    """Wskazuje komórki, w których tworzy się wiązka rozpędzonych elektronów.

    Dla każdej komórki liczymy, jaka część wagi elektronów przypada na te
    rozpędzone. Jeśli przekracza próg, uznajemy komórkę za należącą do wiązki.
    Zwraca maskę takich komórek oraz sam udział w każdej komórce.
    """
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
    """Dzieli elektrony w komórkach wiązki, dopóki nie zbierze się ich dość.

    W każdej takiej komórce dzielimy najcięższe czastki na dwie o połowie wagi.
    Nowa czastka dziedziczy prędkość rodzica i staje tuż obok niego.
    """
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
        # Ile czastek jeszcze chcemy dorobić w tej komórce.
        deficit = cfg.apr_split_target_ppc - counts[c]
        # Dzielimy najpierw najcięższe, bo one najbardziej poprawiają statystykę.
        wsel = electrons.w[in_cell]
        order = in_cell[np.argsort(-wsel)]
        n_to_split = min(deficit, order.size)
        chosen = order[:n_to_split]
        # Bierzemy tylko te, których waga pozwala jeszcze na podział.
        chosen = chosen[electrons.w[chosen] > 2.0 * w_min]
        if chosen.size == 0:
            continue
        half = electrons.w[chosen] * 0.5
        electrons.w[chosen] = half           # rodzic zatrzymuje połowę wagi
        # Dziecko dostaje tę samą prędkość i staje odrobinę z boku, żeby oba
        # nie leżały dokładnie w jednym punkcie.
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
    """Łączy pary elektronów w spokojnych komórkach, gdzie jest ich za dużo.

    Łączymy tylko poza wiązką i tylko czastki o zbliżonej prędkości, żeby
    połączenie jak najmniej zmieniało obraz. Powstała czastka przejmuje sumę
    wag, a jej prędkość jest średnią ważoną obu czastek.
    """
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

        # Sortujemy po prędkości wzdłuż osi, żeby łączyć czastki podobne.
        vx = electrons.vx[in_cell]
        order = in_cell[np.argsort(vx)]
        
        # Łączenie kolejnych par, dopóki nie zejdziemy z nadmiarem.
        n_pairs = n_excess
        k = 0
        for p in range(0, order.size - 1, 2):
            if k >= n_pairs:
                break
            
            a, b = order[p], order[p + 1]
            wa, wb = electrons.w[a], electrons.w[b]
            wsum = wa + wb

            if wsum <= 0:
                continue

            # Prędkość ważona wagą zachowuje masę i pęd obu czastek.
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
    """Wykonuje pełny cykl: najpierw łączenie w tle, potem dzielenie w wiązce."""
    n_merged = coarsen_merge(electrons, cfg, rng)
    n_split = refine_split(electrons, cfg, w_ref, rng)
    return n_split, n_merged
