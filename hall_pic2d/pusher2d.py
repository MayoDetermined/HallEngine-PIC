"""Przenoszenie informacji między siatką a czastkami oraz ruch czastek, na płaszczyźnie.

To odpowiednik operacji z wersji jednowymiarowej, tylko że czastka rozkłada się
teraz na cztery najbliższe węzły, bo leży wewnątrz kwadratu siatki. Rozkładanie
ładunku napisane jest tak, by dobrze radziło sobie z setkami tysięcy czastek,
korzystając z szybkiego zliczania po numerach węzłów. Pchacz czastek przyjmuje
pole magnetyczne o dowolnym kierunku, dzięki czemu obsługuje obie płaszczyzny
tym samym kodem.
"""

import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON


def cic_weights(sp, cfg):
    """Zwraca cztery najbliższe węzły każdej czastki wraz z ich udziałami.

    Ten sam wynik przydaje się dwa razy w jednym kroku, przy rozkładaniu
    ładunku i przy odczycie pola, bo oba dzieją się przed przesunięciem
    czastek, czyli przy tych samych położeniach. Dlatego warto go zapamiętać.
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
    # Numery czterech narożnych węzłów, zapisane jednym wskaźnikiem na węzeł.
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
    """Sumuje na węzłach udziały czastek rozłożone na cztery narożniki."""
    out = np.zeros(n_total)
    for idx, wt in zip(idxs, wts):
        out += np.bincount(idx, weights=values * wt, minlength=n_total)
    return out


def deposit_charge(species_list, cfg, cached=None):
    """Rozkłada ładunek wszystkich czastek na węzły płaszczyzny.

    Jeśli mamy już gotowe najbliższe węzły i udziały z wcześniejszego
    wywołania, można je podać, żeby nie liczyć ich drugi raz.
    """
    n_total = cfg.n1_nodes * cfg.n2_nodes
    rho_flat = np.zeros(n_total)
    cell_vol = cfg.h1 * cfg.h2          # pole komórki, zamienia wagę na gęstość
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
    """Rozkłada na węzły samą liczebność czastek jednego gatunku, do podglądu."""
    n_total = cfg.n1_nodes * cfg.n2_nodes
    if sp.N == 0:
        return np.zeros((cfg.n1_nodes, cfg.n2_nodes))
    idxs, wts = cic_weights(sp, cfg)
    vals = sp.aw / (cfg.h1 * cfg.h2)
    n = _accumulate(idxs, wts, vals, n_total).reshape(cfg.n1_nodes, cfg.n2_nodes)
    _fix_boundary_nodes(n, cfg)
    return n


def _fix_boundary_nodes(arr, cfg):
    """Poprawia węzły brzegowe, które obejmują tylko pół komórki.

    Ich wartości podwajamy, żeby wyszły w tej samej skali co w środku. Kierunku
    zawijającego się w sobie ta poprawka nie dotyczy, bo nie ma tam brzegu.
    """
    arr[0, :] *= 2.0
    arr[-1, :] *= 2.0
    if cfg.x2_bc != "periodic":
        arr[:, 0] *= 2.0
        arr[:, -1] *= 2.0


def gather_field(sp, F1, F2, cfg, cached=None):
    """Odczytuje obie składowe pola w miejscu każdej czastki.

    Wartość między węzłami wyznaczamy jako średnią ważoną z czterech
    otaczających węzłów.
    """
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
    """Przesuwa czastki o jeden krok, przyjmując pole magnetyczne o dowolnym kierunku.

    Tok jest ten sam co w wersji jednowymiarowej: pole elektryczne rozpędza,
    pole magnetyczne obraca prędkość, znów rozpędza. Pole elektryczne leży w
    płaszczyźnie, a pole magnetyczne bierzemy z konfiguracji, więc może być
    skierowane w płaszczyźnie albo poza nią.
    """
    if sp.N == 0:
        return
    dt = cfg.dt
    qm = sp.charge / sp.mass
    b1, b2, b3 = cfg.B_vector(sp.ax1)

    half = 0.5 * dt * qm
    # Pierwsza połowa rozpędzania polem elektrycznym.
    vm1 = sp.av1 + half * E1p
    vm2 = sp.av2 + half * E2p
    vm3 = sp.av3.copy()

    # Wektor opisujący obrót w polu magnetycznym.
    t1 = half * b1
    t2 = half * b2
    t3 = half * b3
    tsq = t1*t1 + t2*t2 + t3*t3
    s = 2.0 / (1.0 + tsq)
    s1, s2, s3 = s*t1, s*t2, s*t3

    # Pomocniczy krok obrotu.
    p1 = vm1 + (vm2*t3 - vm3*t2)
    p2 = vm2 + (vm3*t1 - vm1*t3)
    p3 = vm3 + (vm1*t2 - vm2*t1)

    # Dopełnienie obrotu.
    vp1 = vm1 + (p2*s3 - p3*s2)
    vp2 = vm2 + (p3*s1 - p1*s3)
    vp3 = vm3 + (p1*s2 - p2*s1)

    # Druga połowa rozpędzania polem elektrycznym.
    sp.v1[:sp.N] = vp1 + half * E1p
    sp.v2[:sp.N] = vp2 + half * E2p
    sp.v3[:sp.N] = vp3

    # Przesunięcie po płaszczyźnie zgodnie z nową prędkością.
    sp.x1[:sp.N] = sp.ax1 + sp.av1 * dt
    sp.x2[:sp.N] = sp.ax2 + sp.av2 * dt


def apply_boundaries(sp, cfg):
    """Obsługuje brzegi zależnie od wybranej płaszczyzny.

    Wzdłuż osi kanału czastki, które wyszły poza anodę lub katodę, są
    pochłaniane w obu układach. W drugim kierunku, gdy zawija się on w sobie,
    czastki po prostu wracają z drugiej strony, a gdy ograniczają go ścianki,
    są przy nich pochłaniane. Zwracamy ładunek zebrany na anodzie i katodzie,
    liczbę czastek na katodzie oraz liczbę utraconych na ściankach.
    """
    if sp.N == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Drugi kierunek płaszczyzny.
    wall_w = 0.0
    if cfg.x2_bc == "periodic":
        sp.x2[:sp.N] = np.mod(sp.ax2, cfg.L2)
        wall_kill = np.zeros(sp.N, dtype=bool)
    else:
        wall_kill = (sp.ax2 < 0.0) | (sp.ax2 > cfg.L2)
        if np.any(wall_kill):
            wall_w = float(np.sum(sp.aw[wall_kill]))

    # Oś kanału.
    hit_anode = sp.ax1 < 0.0
    hit_cath = sp.ax1 > cfg.L1
    q_anode = sp.charge * float(np.sum(sp.aw[hit_anode])) if np.any(hit_anode) else 0.0
    q_cath = sp.charge * float(np.sum(sp.aw[hit_cath])) if np.any(hit_cath) else 0.0
    w_cath = float(np.sum(sp.aw[hit_cath])) if np.any(hit_cath) else 0.0

    sp.remove_mask(hit_anode | hit_cath | wall_kill)
    return q_anode, q_cath, w_cath, wall_w


def inject_cathode_electrons(electrons, w_ref, inject_weight, cfg, rng):
    """Dosyła świeże elektrony z katody w głąb kanału, rozłożone równo w poprzek."""
    if inject_weight <= 0.0:
        return
    n_new = int(round(inject_weight / w_ref))
    if n_new <= 0:
        return
    vth = np.sqrt(E_CHARGE * cfg.Te_cathode_eV / M_ELECTRON)
    v1 = -np.abs(rng.normal(0.0, vth, n_new))     # zawsze w głąb kanału
    v2 = rng.normal(0.0, vth, n_new)
    v3 = rng.normal(0.0, vth, n_new)
    x1 = cfg.L1 - 1e-6 - rng.random(n_new) * (cfg.h1 * 0.5)
    x2 = rng.random(n_new) * cfg.L2               # równomiernie w poprzek
    electrons.add(x1, x2, v1, v2, v3, np.full(n_new, w_ref))
