"""Dzielenie i łączenie czastek w wersji dwuwymiarowej, napisane szybko.

Pomysł jest ten sam co w wersji jednowymiarowej: dzielimy czastki tam, gdzie
tworzy się wiązka, a łączymy w spokojnym tle. Komórek jest jednak dużo więcej,
bo cała płaszczyzna, więc zwykła pętla po komórkach byłaby za wolna. Zamiast
niej sortujemy czastki tak, że te z jednej komórki lądują obok siebie, a potem
każdej czastce nadajemy numer porządkowy wewnątrz jej komórki. Mając ten numer,
wybór na przykład kilku pierwszych czastek w każdej komórce to już jedno
porównanie na całych tablicach naraz. Przy dzieleniu bierzemy najpierw
najcięższe czastki, a przy łączeniu grupujemy czastki o zbliżonej prędkości,
żeby połączenie jak najmniej zmieniało obraz.
"""

import numpy as np

from hall_pic.collisions import isotropic_unit_vectors


def _cell_idx(sp, cfg):
    """Zwraca numer komórki na płaszczyźnie, w której leży każda czastka."""
    i = np.clip(np.floor(sp.ax1 / cfg.h1).astype(np.int64), 0, cfg.N1 - 1)
    j = np.floor(sp.ax2 / cfg.h2).astype(np.int64)
    if cfg.x2_bc == "periodic":
        j = np.mod(j, cfg.N2)
    else:
        np.clip(j, 0, cfg.N2 - 1, out=j)
    return i * cfg.N2 + j


def flag_beam_cells(sp, cfg, cell=None):
    """Wskazuje komórki, w których tworzy się wiązka rozpędzonych elektronów.

    Komórkę uznajemy za należącą do wiązki, gdy udział wagi elektronów powyżej
    progu przekracza zadaną wartość. Zwraca maskę takich komórek oraz sam
    udział w każdej komórce.
    """
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
    """Nadaje każdej czastce numer porządkowy wewnątrz jej komórki, w zadanej kolejności."""
    sc = cell[order]
    starts = np.searchsorted(sc, np.arange(cfg.n_cells), side="left")
    rank = np.arange(sc.size) - starts[sc]
    return sc, rank


def refine_split(electrons, cfg, w_ref, rng):
    """Dzieli najcięższe elektrony w komórkach wiązki, aż zbierze się ich dość."""
    if electrons.N == 0 or electrons.N >= cfg.max_particles:
        return 0                     # twardy sufit chroni przed lawiną podziałów
    cell = _cell_idx(electrons, cfg)
    flags, _ = flag_beam_cells(electrons, cfg, cell)
    if not np.any(flags):
        return 0
    counts = np.bincount(cell, minlength=cfg.n_cells)
    deficit = np.where(flags, np.maximum(cfg.apr_split_target_ppc - counts, 0), 0)
    if deficit.sum() == 0:
        return 0

    w_min = cfg.apr_min_weight_ratio * w_ref
    # Grupujemy czastki po komórkach, a w komórce najcięższe stawiamy pierwsze.
    order = np.lexsort((-electrons.aw, cell))
    sc, rank = _ranks_in_cells(cell, order, cfg)
    take = (rank < deficit[sc]) & (electrons.aw[order] > 2.0 * w_min)
    chosen = order[take]
    if chosen.size == 0:
        return 0

    # Dane odczytujemy przed dopisaniem nowych czastek, bo dopisanie może
    # przenieść tablice w inne miejsce w pamięci.
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

    electrons.w[chosen] = half              # rodzic zatrzymuje połowę wagi
    electrons.add(x1, x2, v1, v2, v3, half.copy())
    return int(chosen.size)


def coarsen_merge(electrons, cfg, rng):
    """Łączy czwórki elektronów w pary, nie tracąc masy, pędu ani energii.

    Z każdej czwórki robimy dwie czastki o połowie łącznej wagi. Obie mają
    prędkość równą średniej prędkości czwórki, ale rozsuniętą symetrycznie o
    tyle w losowym kierunku, by zachować także średnią szybkość. Dzięki temu
    zgadza się i łączna waga, i pęd, i energia grupy.

    Prostsze łączenie dwóch czastek w jedną przez samo uśrednienie prędkości
    zachowałoby masę i pęd, ale gubiłoby energię, sztucznie studząc tło, więc
    tu z niego nie korzystamy. Czwórka zamieniona w parę usuwa dwie czastki,
    czyli redukuje ich liczbę tak samo skutecznie jak łączenie parami.
    """
    if electrons.N == 0:
        return 0
    cell = _cell_idx(electrons, cfg)
    flags, _ = flag_beam_cells(electrons, cfg, cell)
    counts = np.bincount(cell, minlength=cfg.n_cells)
    over = (~flags) & (counts > cfg.apr_max_ppc)
    excess = np.where(over, counts - cfg.apr_max_ppc, 0)
    # Każda czwórka usuwa dwie czastki, stąd taki dobór liczby czwórek.
    n_groups = np.where(over, np.minimum(excess // 2, counts // 4), 0)
    total_groups = int(n_groups.sum())
    if total_groups == 0:
        return 0

    order = np.lexsort((electrons.av1, cell))   # w komórce układamy po prędkości, by łączyć podobne
    sc, rank = _ranks_in_cells(cell, order, cfg)
    part = rank < 4 * n_groups[sc]
    if not np.any(part):
        return 0

    idx = order[part]
    # Nadajemy każdej czastce wspólny numer jej czwórki.
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

    # Z każdej czwórki zostają dwie czastki, rozsunięte symetrycznie wokół
    # średniej prędkości. Pozostałe dwie znikają.
    for s, sign in ((0, +1.0), (1, -1.0)):
        m = slot == s
        ii = idx[m]
        g = gid[m]
        electrons.w[ii] = 0.5 * W[g]
        electrons.v1[ii] = u1[g] + sign * d[g] * e1[g]
        electrons.v2[ii] = u2[g] + sign * d[g] * e2[g]
        electrons.v3[ii] = u3[g] + sign * d[g] * e3[g]
    # Położenia czastek, które zostają, nie ruszamy. Uśrednianie ich psułoby
    # ciągłość tam, gdzie kierunek zawija się w sobie.

    kill = np.zeros(electrons.N, dtype=bool)
    kill[idx[slot >= 2]] = True
    electrons.remove_mask(kill)
    return 2 * total_groups


def run_apr(electrons, cfg, w_ref, rng):
    """Wykonuje pełny cykl: najpierw łączenie w tle, potem dzielenie w wiązce."""
    n_merged = coarsen_merge(electrons, cfg, rng)
    n_split = refine_split(electrons, cfg, w_ref, rng)
    return n_split, n_merged
