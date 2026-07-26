"""Przenoszenie informacji między siatką a czastkami oraz ruch czastek.

Tu mieszczą się wszystkie kroki, które łączą świat siatki ze światem czastek:
rozkładanie ładunku czastek na węzły, odczytywanie pola w miejscu czastki,
przesuwanie czastek pod wpływem pól, a także to, co dzieje się, gdy czastka
dotrze do anody albo katody. Choć śledzimy tylko położenie wzdłuż osi kanału,
prędkość ma pełne trzy składowe. Pole elektryczne działa wzdłuż osi, a pole
magnetyczne jest skierowane w poprzek, i to właśnie ich złożenie nadaje
elektronom charakterystyczny dryf typowy dla silnika Halla.
"""

import numpy as np

from .constants import E_CHARGE


def deposit_charge(species_list, cfg):
    """Rozkłada ładunek wszystkich czastek na węzły siatki.

    Każda czastka dokłada się do dwóch najbliższych węzłów tym więcej, im bliżej
    danego węzła leży. Zwraca gęstość ładunku na wszystkich węzłach.
    """
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
        contrib = sp.charge * sp.aw / dx
        # Dokładamy udział do węzła po lewej i po prawej stronie czastki.
        np.add.at(rho, i, contrib * (1.0 - frac))
        np.add.at(rho, i + 1, contrib * frac)
    # Węzły na samych brzegach obejmują tylko pół komórki, więc ich gęstość
    # trzeba podwoić, żeby wyszła w tej samej skali co w środku.
    rho[0] *= 2.0
    rho[-1] *= 2.0
    return rho


def deposit_number_density(sp, cfg):
    """Rozkłada na węzły samą liczebność czastek jednego gatunku.

    Działa tak samo jak rozkładanie ładunku, tylko bez mnożenia przez ładunek.
    Przydaje się do podglądu gęstości elektronów albo jonów.
    """
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
    """Odczytuje pole elektryczne w miejscu każdej czastki.

    Wartość między węzłami wyznaczamy jako średnią ważoną z dwóch sąsiednich
    węzłów, tak samo jak przy rozkładaniu ładunku, tylko w drugą stronę.
    """
    dx = cfg.dx
    fi = sp.ax / dx
    i = np.clip(np.floor(fi).astype(np.int64), 0, cfg.Nx - 1)
    frac = fi - i
    return E_nodes[i] * (1.0 - frac) + E_nodes[i + 1] * frac


def boris_push(sp, Ex_p, cfg):
    """Przesuwa czastki o jeden krok czasu pod wpływem pól.

    Najpierw pole elektryczne rozpędza czastkę wzdłuż osi, potem pole
    magnetyczne obraca jej prędkość, a na końcu pole elektryczne działa jeszcze
    raz. Taki podział to sprawdzony sposób, który dobrze zachowuje energię.
    Siła magnetyczna zależy od miejsca, bo pole jest silniejsze przy wylocie.
    """
    if sp.N == 0:
        return
    dt = cfg.dt
    qm = sp.charge / sp.mass
    By = cfg.B_profile(sp.ax)          # natężenie pola tam, gdzie akurat jest czastka

    # Pierwsza połowa rozpędzania polem elektrycznym, tylko wzdłuż osi.
    vminus_x = sp.avx + qm * Ex_p * 0.5 * dt
    vminus_y = sp.avy.copy()
    vminus_z = sp.avz.copy()

    # Obrót prędkości wokół kierunku pola magnetycznego.
    ty = qm * By * 0.5 * dt
    s_fac = 2.0 * ty / (1.0 + ty * ty)

    # Pomocniczy krok obrotu: dokładamy do prędkości jej iloczyn wektorowy z
    # wektorem obrotu. Pole leży wzdłuż jednego kierunku, więc zmieniają się
    # tylko dwie składowe.
    vprime_x = vminus_x + (-vminus_z * ty)
    vprime_y = vminus_y
    vprime_z = vminus_z + (vminus_x * ty)

    # Dopełnienie obrotu drugim iloczynem wektorowym.
    vplus_x = vminus_x + (-vprime_z * s_fac)
    vplus_y = vminus_y
    vplus_z = vminus_z + (vprime_x * s_fac)

    # Druga połowa rozpędzania polem elektrycznym.
    sp.vx[:sp.N] = vplus_x + qm * Ex_p * 0.5 * dt
    sp.vy[:sp.N] = vplus_y
    sp.vz[:sp.N] = vplus_z

    # Na koniec przesuwamy czastkę wzdłuż osi zgodnie z jej nową prędkością.
    sp.x[:sp.N] = sp.ax + sp.avx * dt


def apply_boundaries(sp, cfg):
    """Pochłania czastki, które opuściły kanał przez anodę lub katodę.

    Sumuje ładunek, który przy tym trafił na każdą z elektrod, i zwraca go
    razem z liczbą jonów, które dobiły do katody. Te wielkości potrzebne są
    później do zasilania obwodu i do dosyłania elektronów z katody.
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
    """Dosyła świeże elektrony z katody w głąb kanału.

    Ile ich dosłać, mówi podana łączna waga, zwykle powiązana ze strumieniem
    jonów uciekających do katody. Nowe elektrony startują tuż przy katodzie i
    lecą do środka, z losowymi prędkościami o zadanej temperaturze.
    """
    if n_inject_weight <= 0.0:
        return
    n_new = int(round(n_inject_weight / w_ref))
    if n_new <= 0:
        return
    from .constants import M_ELECTRON, K_BOLTZMANN
    vth = np.sqrt(E_CHARGE * cfg.Te_cathode_eV / M_ELECTRON)
    # Prędkości losowe, przy czym wzdłuż osi zawsze skierowane w głąb kanału.
    vx = -np.abs(rng.normal(0.0, vth, n_new))
    vy = rng.normal(0.0, vth, n_new)
    vz = rng.normal(0.0, vth, n_new)
    # Startujemy tuż przy samej katodzie.
    x0 = cfg.L - 1e-6 - rng.random(n_new) * (cfg.dx * 0.5)
    electrons.add(x0, vx, vy, vz, np.full(n_new, w_ref))
