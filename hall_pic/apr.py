"""Adaptacyjne dzielenie i łączenie czastek dla lepszej statystyki wiązki.

Wiązka rozpędzonych elektronów to niewielka mniejszość, więc łatwo o nią
statystycznie zubożeć. Dlatego tam, gdzie taka wiązka się tworzy, dzielimy
czastki na drobniejsze, żeby było ich więcej i żeby obraz był gładszy. Z kolei
w spokojnym tle, gdzie czastek robi się za dużo, łączymy je z powrotem, żeby
obliczenia nie spowalniały. Zarówno dzielenie, jak i łączenie zachowuje masę,
pęd oraz energię, więc nie zaburza fizyki, a jedynie zmienia liczbę czastek.
Łączenia nie schodzimy poniżej pewnej najmniejszej wagi, żeby liczba czastek
nie rosła bez opamiętania.
"""

import numpy as np

from .constants import E_CHARGE
from .collisions import isotropic_unit_vectors


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
    """Łączy czwórki elektronów w pary, nie tracąc masy, pędu ani energii.

    Łączymy tylko poza wiązką i tylko czastki o zbliżonej prędkości, żeby jak
    najmniej zmieniać obraz. Z każdej czwórki robimy dwie czastki o połowie
    łącznej wagi. Obie mają prędkość równą średniej prędkości czwórki, ale
    rozsuniętą symetrycznie o tyle w losowym kierunku, by zachować także średnią
    szybkość. Dzięki temu zgadza się i łączna waga, i pęd, i energia grupy.

    Wcześniej łączyliśmy czastki parami, uśredniając ich prędkość. Zachowywało
    to masę i pęd, ale gubiło energię i sztucznie studziło tło, co zaniżało
    udział rozpędzonych elektronów. Dlatego teraz stosujemy łączenie czwórek.
    """
    if electrons.N == 0:
        return 0
    ci = _cell_index(electrons.ax, cfg)
    flags, _ = flag_beam_cells(electrons, cfg)
    counts = np.bincount(ci, minlength=cfg.Nx)
    over = (~flags) & (counts > cfg.apr_max_ppc)
    excess = np.where(over, counts - cfg.apr_max_ppc, 0)
    # Każda czwórka usuwa dwie czastki, stąd taki dobór liczby czwórek.
    n_groups = np.where(over, np.minimum(excess // 2, counts // 4), 0)
    total_groups = int(n_groups.sum())
    if total_groups == 0:
        return 0

    # Grupujemy czastki po komórkach, a w komórce układamy je po prędkości,
    # żeby łączyć podobne.
    order = np.lexsort((electrons.avx, ci))
    sc = ci[order]
    starts = np.searchsorted(sc, np.arange(cfg.Nx), side="left")
    rank = np.arange(sc.size) - starts[sc]
    part = rank < 4 * n_groups[sc]
    if not np.any(part):
        return 0

    idx = order[part]
    # Nadajemy każdej czastce wspólny numer jej czwórki.
    goff = np.concatenate(([0], np.cumsum(n_groups)[:-1]))
    gid = goff[sc[part]] + (rank[part] // 4)
    slot = rank[part] % 4

    w = electrons.w[idx]
    vx = electrons.vx[idx]; vy = electrons.vy[idx]; vz = electrons.vz[idx]
    W = np.bincount(gid, weights=w, minlength=total_groups)
    Px = np.bincount(gid, weights=w*vx, minlength=total_groups)
    Py = np.bincount(gid, weights=w*vy, minlength=total_groups)
    Pz = np.bincount(gid, weights=w*vz, minlength=total_groups)
    Es = np.bincount(gid, weights=w*(vx*vx + vy*vy + vz*vz), minlength=total_groups)

    Wsafe = np.maximum(W, 1e-300)
    ux, uy, uz = Px/Wsafe, Py/Wsafe, Pz/Wsafe
    d2 = np.maximum(Es/Wsafe - (ux*ux + uy*uy + uz*uz), 0.0)
    d = np.sqrt(d2)
    ex, ey, ez = isotropic_unit_vectors(total_groups, rng)

    # Z każdej czwórki zostają dwie czastki, rozsunięte symetrycznie wokół
    # średniej prędkości. Pozostałe dwie znikają.
    for s, sign in ((0, +1.0), (1, -1.0)):
        m = slot == s
        ii = idx[m]
        g = gid[m]
        electrons.w[ii] = 0.5 * W[g]
        electrons.vx[ii] = ux[g] + sign * d[g] * ex[g]
        electrons.vy[ii] = uy[g] + sign * d[g] * ey[g]
        electrons.vz[ii] = uz[g] + sign * d[g] * ez[g]
    # Położenia czastek, które zostają, nie ruszamy.

    kill = np.zeros(electrons.N, dtype=bool)
    kill[idx[slot >= 2]] = True
    electrons.remove_mask(kill)
    return 2 * total_groups


def run_apr(electrons, cfg, w_ref, rng):
    """Wykonuje pełny cykl: najpierw łączenie w tle, potem dzielenie w wiązce."""
    n_merged = coarsen_merge(electrons, cfg, rng)
    n_split = refine_split(electrons, cfg, w_ref, rng)
    return n_split, n_merged
