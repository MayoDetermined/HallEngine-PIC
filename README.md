# PIC 1D3V  silnik Halla (SPT-100) z obwodem RLC, null-MCC i APR

> **Wersja 2D3V** (geometrie osiowo-azymutalna z-θ oraz osiowo-promieniowa z-r)
> znajduje się w `hall_pic2d/`, uruchamiana przez `run2d.py`.
> Dokumentacja: [README_2D.md](README_2D.md).
>
> **Benchmark LANDMARK** i kalibracja przekrojów czynnych: `landmark/`,
> uruchamiany przez `python -m landmark.benchmark`.
> Dokumentacja: [README_LANDMARK.md](README_LANDMARK.md).
> **Uwaga:** przekroje czynne zostały skalibrowane względem danych LANDMARK,
> więc wyniki sprzed kalibracji nie są odtwarzalne tym kodem.

Elektrostatyczna symulacja Particle-in-Cell w geometrii osiowej 1D (prędkości 3D)
dla silnika Halla typu SPT-100. Model pokazuje samouzgodnione sprzężenie plazmy z
zewnętrznym **obwodem RLC** (cel ~4 A), zderzenia metodą **null-collision MCC** z
**zamrożonymi neutralami**, adaptacyjne zagęszczanie superczątstek (**APR**) dla
lepszej statystyki **wiązek elektronów uciekających (RE, runaway)** oraz **podgląd
na żywo** w Matplotlib.

## Szybki start

```bash
pip install -r requirements.txt
python run.py                      # podgląd na żywo, do 200 ns
```

Inne tryby:

```bash
python run.py --t-end 1e-6         # dociągnij do 1 mikrosekundy
python run.py --headless out/      # bez okna: klatki PNG do out/
python run.py --no-apr             # wyłącz APR (porównanie dokładności ogona RE)
python run.py --steps 4000         # ogranicz liczbę kroków (szybki test)
python run.py --nx 512 --ppc 200   # gęstsza siatka / więcej cząstek
```

## Co obejmuje model

| Wymaganie                        | Realizacja |
|----------------------------------|------------|
| Geometria SPT-100                | `Config`: L=2.5 cm, anoda x=0, katoda x=L, B_r gaussowski ~180 G @ 20 mm |
| Obwód 4 A (RLC z kondensatorem)  | `circuit.py`: RK4 dla `L dI/dt = V_ps−R·I−V_C`, `C dV_C/dt = I_L−I_d`; V_C = potencjał anody (BC Poissona) |
| Ważenie/normalizacja superczątstek | `species.py`: waga `w [1/m²]`, spójna dla ładunku (Poisson) i prądu (`I = q·A·Σw/dt`) |
| Zamrożone neutrale               | `Config.neutral_density(x)`  stały profil, brak dynamiki; cel dla kolizji |
| null-MCC + losowy obrót wektora  | `collisions.py`: `nu_max` stałe, losowany podzbiór, izotropowe rozproszenie na sferze |
| Jonizacja (pary e⁻/jon)          | `collide_electrons`: próg 12.13 eV, podział energii, nowy elektron + jon |
| APR dla wiązek RE                | `apr.py`: detekcja komórek wiązkowych + split (zagęszczanie) / merge (rozrzedzanie) |
| Wiązki RE                        | pchacz Borisa (E_x, B_y), ogon EEDF > `E_RE`, prąd `I_RE`, smuga w przestrzeni fazowej |
| Czas ns → µs                     | `dt=5 ps`; `--t-end 1e-6` dla pełnego µs |
| Podgląd na żywo                  | `diagnostics.py`: 6 paneli Matplotlib, odświeżanych co `plot_interval` |

## Schemat kroku czasowego (leapfrog / Boris)

1. Depozycja ładunku `ρ` (CIC 1. rzędu) → 2. Poisson `φ, E` (BC: `φ(0)=V_C`) →
3. zbieranie `E` do cząstek + pchacz Borisa (E_x, B_y) → 4. warunki brzegowe
(absorpcja anoda/katoda, akumulacja `I_d`) → 5. emisja elektronów z katody →
6. kolizje null-MCC (e: elast./wzbudz./jonizacja; jon: CEX/elast.) →
7. APR (co `apr_interval`) → 8. całkowanie obwodu RLC → 9. diagnostyka.

## Fizyka wiązek RE

Elektrony wpadające w obszar silnego pola `E_x` przy warstwie przyanodowej
przyspieszają szybciej niż tracą energię na zderzenia (neutrale zamrożone i
rozrzedzone) → „uciekają" w energii, tworząc **wąską smugę w przestrzeni fazowej
(x, v_x)** i **ogon w EEDF** powyżej progu `E_RE` (domyślnie 60 eV). Panel 1
pokazuje formowanie smugi na żywo; panel 5  narastanie ogona; `I_RE` w panelu 4
kwantyfikuje prąd niesiony przez populację RE.

APR adaptacyjnie **zagęszcza** superczątstki w komórkach, gdzie frakcja RE
przekracza próg, więc ogon rozkładu jest próbkowany dużo gęściej niż w zwykłym
PIC ze stałą wagą  bez wysadzania kosztu w obszarze tła (tam działa merge).

## Kluczowe parametry (plik `hall_pic/config.py`)

- **Geometria/pole B:** `L, Nx, A_channel, B_max, x_B, sigma_B`
- **Neutrale:** `n_n_anode, n_n_exit, T_neutral` (zamrożone)
- **Obwód:** `V_ps, R_circ, L_circ, C_circ, V_C_init, I_L_init`
- **Plazma:** `n0_plasma, Te0_eV, Ti0_eV, n_ppc`
- **APR:** `E_RE_eV, apr_beam_frac_threshold, apr_split_target_ppc, apr_max_ppc`
- **Czas:** `dt, t_end, circuit_substeps`
- **Podgląd:** `live_view, plot_interval, headless_save_dir`

### Strojenie pod wyraźniejsze RE
- podnieś `V_ps`/`V_C_init` (silniejsze pole przyspieszające),
- obniż `n_n_*` (mniej zderzeń hamujących → więcej ucieczek),
- obniż `E_RE_eV` by zaliczać do RE niższe energie,
- zwiększ `apr_split_target_ppc` dla gładszego ogona.

## Ważne uproszczenia (do świadomego rozszerzenia)

- **Przekroje czynne** (`cross_sections.py`) to gładkie fity inżynierskie dla Xe.
  Do zastosowań ilościowych podstaw dane tablicowe (LXCat/Biagi).
- **Sprzężenie z obwodem** używa prądu przewodzenia na anodzie (`I_d = −Q_anoda·A/dt`);
  pełna zgodność ładunkowa wymagałaby członu prądu przesunięcia (metoda
  Vahedi–DiPeso). Wystarczające do transientu i dynamiki RLC.
- **Merge w APR** zachowuje masę i pęd (nie energię)  stosowany tylko poza
  obszarem wiązki, gdzie ogon nie jest badany.
- **Emisja katody** to prosty model neutralizatora proporcjonalny do strumienia
  jonów; nie zawiera pełnej fizyki katody wnękowej.
- Model **1D**: brak strat na ścianach promieniowych i geometrii azymutalnej
  (choć dryf ExB w kierunku azymutalnym jest w pchaczu Borisa obecny).

## Struktura kodu

```
run.py                  # CLI + pętla główna
requirements.txt
hall_pic/
  config.py             # wszystkie parametry (dataclass Config)
  constants.py          # stałe fizyczne SI
  cross_sections.py     # przekroje czynne Xe (e i jony)
  species.py            # kontener superczątstek (SoA) + ważenie
  poisson.py            # solver Poissona 1D (prekomputowana odwrotność)
  circuit.py            # obwód RLC (RK4)
  pusher.py             # depozycja/zbieranie/Boris/brzegi/emisja
  collisions.py         # null-collision MCC + izotropowe rozpraszanie
  apr.py                # adaptive particle refinement (split/merge)
  diagnostics.py        # podgląd na żywo (6 paneli) / zapis PNG
  simulation.py         # klasa Simulation spinająca krok czasowy

run2d.py                # CLI wersji 2D3V (--geometry z-theta | z-r)
hall_pic2d/             # rdzeń 2D3V  patrz README_2D.md
  config2d.py           # Config2D z przełącznikiem geometrii
  poisson2d.py          # separowalny solver 2D (FFT / DST + Thomas)
  species2d.py          # superczątstki 2D3V
  pusher2d.py           # CIC dwuliniowa, Boris z pełnym 3-wektorem B
  collisions2d.py       # null-MCC w 2D
  apr2d.py              # APR zwektoryzowany (lexsort + searchsorted)
  diagnostics2d.py      # mapy 2D + przestrzeń fazowa + obwód
  simulation2d.py       # pętla główna 2D
```

Moduły `constants.py`, `cross_sections.py` i `circuit.py` są współdzielone
przez obie wersje  fizyka zderzeń i obwód nie zależą od wymiarowości.

## Wydajność

Wektoryzowany NumPy; ~400–600 kroków/s (Nx=256, ~30k cząstek makro).
1 µs ≈ 200 000 kroków ≈ 8–10 min. Dla dużych przebiegów użyj `--headless`
(szybszy zapis niż interaktywne okno). Możliwe dalsze przyspieszenie: Numba/`@njit`
na pchaczu i depozycji, albo `scipy.sparse` na Poissonie dla dużych `Nx`.
