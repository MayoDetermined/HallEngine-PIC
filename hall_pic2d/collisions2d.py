"""Zderzenia czastek z gazem w wersji dwuwymiarowej.

Cała fizyka jest taka sama jak w wersji jednowymiarowej. Zmienia się tylko
tyle, że czastka ma dwie współrzędne położenia, a gęstość gazu odczytujemy po
położeniu wzdłuż osi kanału. Po każdym zderzeniu prędkość czastki obracamy w
losowym kierunku.
"""

import numpy as np

from hall_pic.constants import E_CHARGE, M_ELECTRON, K_BOLTZMANN
from hall_pic import cross_sections as xs
from hall_pic.collisions import isotropic_unit_vectors


class NullCollisionMCC2D:
    """Obsługuje zderzenia elektronów i jonów z gazem, na płaszczyźnie."""

    def __init__(self, cfg, rng):
        """Wyznacza największe częstości zderzeń i szanse losowania dla obu gatunków."""
        self.cfg = cfg
        self.rng = rng
        eps = np.linspace(0.01, cfg.eps_max_grid_eV, cfg.n_energy_grid)
        n_n_max = cfg.n_n_anode

        v_e = xs.speed_from_energy_e(eps)
        self.nu_max_e = float(np.max(n_n_max * xs.sigma_total_e(eps) * v_e))
        self.P_e = 1.0 - np.exp(-self.nu_max_e * cfg.dt)

        v_i = np.sqrt(2.0 * eps * E_CHARGE / cfg.m_ion)
        self.nu_max_i = float(np.max(n_n_max * xs.sigma_total_ion(eps) * v_i))
        self.P_i = 1.0 - np.exp(-self.nu_max_i * cfg.dt)

    def _set_iso(self, sp, gidx, speed, u1, u2, u3):
        """Ustawia prędkość wskazanych czastek na nową długość i losowy kierunek."""
        sp.v1[gidx] = speed * u1
        sp.v2[gidx] = speed * u2
        sp.v3[gidx] = speed * u3

    # Zderzenia elektronów.
    def collide_electrons(self, electrons, ions):
        """Rozgrywa zderzenia elektronów i zwraca, ile powstało przy tym nowych jonów.

        Wylosowane elektrony mogą odbić się sprężyście, wzbudzić atom albo go
        zjonizować. Jonizacja tworzy nową parę: dodatkowy elektron i jon.
        """
        cfg, rng = self.cfg, self.rng
        N = electrons.N
        if N == 0 or self.P_e <= 0.0:
            return 0
        n_cand = rng.binomial(N, self.P_e)
        if n_cand == 0:
            return 0
        idx = rng.choice(N, size=n_cand, replace=False)

        v1 = electrons.v1[idx]; v2 = electrons.v2[idx]; v3 = electrons.v3[idx]
        speed = np.sqrt(v1*v1 + v2*v2 + v3*v3)
        eps = 0.5 * M_ELECTRON * speed**2 / E_CHARGE
        n_n = cfg.neutral_density(electrons.x1[idx])

        inv = 1.0 / self.nu_max_e
        p_el = n_n * xs.sigma_elastic_e(eps) * speed * inv
        p_ex = n_n * xs.sigma_excitation_e(eps) * speed * inv
        p_iz = n_n * xs.sigma_ionization_e(eps) * speed * inv

        r = rng.random(n_cand)
        is_el = r < p_el
        is_ex = (r >= p_el) & (r < p_el + p_ex)
        is_iz = (r >= p_el + p_ex) & (r < p_el + p_ex + p_iz)
        # Kto wylosował więcej niż suma szans, trafił na zderzenie zerowe i nic go nie zmienia.

        if np.any(is_el):
            sel = np.nonzero(is_el)[0]
            dE = 2.0 * M_ELECTRON / cfg.m_ion
            new_eps = np.maximum(eps[sel] * (1.0 - dE), 1e-3)
            u = isotropic_unit_vectors(sel.size, rng)
            self._set_iso(electrons, idx[sel], xs.speed_from_energy_e(new_eps), *u)

        if np.any(is_ex):
            sel = np.nonzero(is_ex)[0]
            new_eps = np.maximum(eps[sel] - xs.E_EXC_XE, 1e-3)
            u = isotropic_unit_vectors(sel.size, rng)
            self._set_iso(electrons, idx[sel], xs.speed_from_energy_e(new_eps), *u)

        n_new = 0
        if np.any(is_iz):
            sel = np.nonzero(is_iz)[0]
            gi = idx[sel]
            avail = np.maximum(eps[sel] - xs.E_ION_XE, 0.0)
            split = rng.random(sel.size)
            e_prim, e_sec = avail * split, avail * (1.0 - split)

            u1 = isotropic_unit_vectors(sel.size, rng)
            self._set_iso(electrons, gi,
                          xs.speed_from_energy_e(np.maximum(e_prim, 1e-3)), *u1)

            wpar = electrons.w[gi].copy()
            xa = electrons.x1[gi].copy()
            xb = electrons.x2[gi].copy()
            u2 = isotropic_unit_vectors(sel.size, rng)
            sp2 = xs.speed_from_energy_e(np.maximum(e_sec, 1e-3))
            electrons.add(xa, xb, sp2*u2[0], sp2*u2[1], sp2*u2[2], wpar)

            vth_i = np.sqrt(K_BOLTZMANN * cfg.T_neutral / cfg.m_ion)
            ions.add(xa.copy(), xb.copy(),
                     rng.normal(0, vth_i, sel.size),
                     rng.normal(0, vth_i, sel.size),
                     rng.normal(0, vth_i, sel.size), wpar.copy())
            n_new = sel.size
        return n_new

    # Zderzenia jonów.
    def collide_ions(self, ions):
        """Rozgrywa zderzenia jonów z atomami: wymianę ładunku i odbicia sprężyste."""
        cfg, rng = self.cfg, self.rng
        N = ions.N
        if N == 0 or self.P_i <= 0.0:
            return
        n_cand = rng.binomial(N, self.P_i)
        if n_cand == 0:
            return
        idx = rng.choice(N, size=n_cand, replace=False)
        v1 = ions.v1[idx]; v2 = ions.v2[idx]; v3 = ions.v3[idx]

        vth_n = np.sqrt(K_BOLTZMANN * cfg.T_neutral / cfg.m_ion)
        n1 = rng.normal(0, vth_n, n_cand)
        n2 = rng.normal(0, vth_n, n_cand)
        n3 = rng.normal(0, vth_n, n_cand)
        g1, g2, g3 = v1 - n1, v2 - n2, v3 - n3
        g = np.sqrt(g1*g1 + g2*g2 + g3*g3)
        eps_rel = 0.5 * cfg.m_ion * g**2 / E_CHARGE

        n_n = cfg.neutral_density(ions.x1[idx])
        inv = 1.0 / self.nu_max_i
        p_cex = n_n * xs.sigma_cex_ion(eps_rel) * g * inv
        p_el = n_n * xs.sigma_elastic_ion(eps_rel) * g * inv

        r = rng.random(n_cand)
        is_cex = r < p_cex
        is_el = (r >= p_cex) & (r < p_cex + p_el)

        if np.any(is_cex):          # przy wymianie ładunku jon przejmuje prędkość atomu
            s = np.nonzero(is_cex)[0]
            ions.v1[idx[s]] = n1[s]; ions.v2[idx[s]] = n2[s]; ions.v3[idx[s]] = n3[s]

        if np.any(is_el):           # przy odbiciu sprężystym obracamy ruch względem wspólnego środka masy
            s = np.nonzero(is_el)[0]
            u1, u2, u3 = isotropic_unit_vectors(s.size, rng)
            gs = g[s]
            ions.v1[idx[s]] = 0.5*(v1[s] + n1[s]) + 0.5*gs*u1
            ions.v2[idx[s]] = 0.5*(v2[s] + n2[s]) + 0.5*gs*u2
            ions.v3[idx[s]] = 0.5*(v3[s] + n3[s]) + 0.5*gs*u3
