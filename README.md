# Symulacja plazmy silnika Halla metodą czastek w komórce

Kinetyczna symulacja wyładowania w silniku Halla, napisana w Pythonie. Program
śledzi pojedyncze czastki plazmy w kanale silnika, samouzgodnione pole
elektryczne, zewnętrzny obwód zasilający oraz zderzenia z gazem roboczym.
Powstał po to, żeby zobaczyć, jak w kanale formuje się wiązka rozpędzonych
elektronów, i pokazać ją na żywo w trakcie liczenia.

Dostępne są dwie wersje: jednowymiarowa, licząca wzdłuż osi kanału, oraz
dwuwymiarowa, licząca na płaszczyźnie i pozwalająca wybrać jedną z dwóch
płaszczyzn kanału.

## Co program potrafi

- Śledzi elektrony i jony ksenonu z pełnym, trójskładnikowym wektorem prędkości.
- Rozwiązuje pole elektryczne z rozkładu ładunku w każdym kroku, bez zakładania
  z góry, że plazma jest obojętna. Dzięki temu widać warstwy przyścienne przy
  elektrodach i ściankach.
- Łączy plazmę z zewnętrznym obwodem zasilającym, w którym zasilacz, opór,
  cewka i kondensator dają docelowo prąd rzędu kilku amperów. Napięcie na
  kondensatorze jest zarazem napięciem anody.
- Liczy zderzenia z zamrożonym gazem metodą zderzeń zerowych, z jonizacją,
  wzbudzeniem i odbiciami. Po każdym zderzeniu kierunek prędkości jest losowany.
- Adaptacyjnie dzieli czastki tam, gdzie tworzy się wiązka, i łączy je w tle,
  żeby poprawić dokładność bez spowalniania obliczeń. Dzielenie i łączenie
  zachowuje masę, pęd i energię.
- Pokazuje przebieg na żywo w oknie z sześcioma wykresami albo zapisuje kolejne
  klatki do plików, gdy pracujemy bez ekranu.

## Wymagania

- Python w wersji 3.10 lub nowszej
- NumPy oraz Matplotlib (patrz `requirements.txt`)
- SciPy jest opcjonalne. Skrypty rysujące mapy korzystają z niego do lekkiego
  wygładzania linii konturowych, ale bez niego również działają.

Instalacja zależności:

```bash
pip install -r requirements.txt
```

## Szybki start

Wersja jednowymiarowa, z podglądem na żywo:

```bash
python run.py
```

Wersja dwuwymiarowa, płaszczyzna wokół kanału:

```bash
python run2d.py
```

Wersja dwuwymiarowa, płaszczyzna w poprzek kanału, ze ściankami:

```bash
python run2d.py --geometry z-r
```

Praca bez okna, z zapisem klatek do katalogu:

```bash
python run.py --headless klatki/
```

## Najważniejsze opcje wiersza poleceń

Obie wersje przyjmują między innymi:

- `--steps` liczba kroków, przydatne do krótkiego przebiegu testowego
- `--t-end` czas końcowy w sekundach, gdy chcemy liczyć do konkretnej chwili
- `--headless KATALOG` praca bez okna, z zapisem klatek
- `--no-apr` bez dzielenia i łączenia czastek
- `--no-collisions` bez zderzeń z gazem
- `--seed` ziarno losowości, żeby przebieg dało się powtórzyć

Wersja dwuwymiarowa dodatkowo:

- `--geometry z-theta` albo `--geometry z-r` wybór płaszczyzny kanału
- `--n1`, `--n2` liczba komórek siatki w obu kierunkach
- `--ppc` liczba czastek modelowych na komórkę

Pełną listę pokazuje `python run.py --help` oraz `python run2d.py --help`.

## Gotowe rysunki podsumowujące

Poza podglądem na żywo można wygenerować gotowe rysunki. Pokazują one, jak
sytuacja zmienia się w czasie, bo wiązka nie powstaje od razu, tylko stopniowo
się formuje.

Zestaw rysunków dla obu wersji:

```bash
python make_figures.py
```

Mapy uśrednionej gęstości z zaznaczoną wiązką i warstwami przyściennymi:

```bash
python make_density_maps.py
```

## Jak to działa w skrócie

Każdy krok w czasie wygląda tak samo. Najpierw rozkładamy ładunek czastek na
siatkę i wyznaczamy z niego potencjał oraz pole, przy czym napięcie anody
bierzemy z obwodu. Potem odczytujemy pole w miejscu każdej czastki i przesuwamy
czastki, obracając ich prędkość w polu magnetycznym. Następnie pochłaniamy te,
które trafiły na elektrody albo ścianki, i z zebranego ładunku odczytujemy prąd
wyładowania. Katoda dosyła świeże elektrony, rozgrywamy zderzenia z gazem, co
pewien czas dzielimy i łączymy czastki, na koniec przesuwamy obwód i odświeżamy
podgląd.

Pole magnetyczne ma kształt dzwonu ze szczytem przy wylocie kanału. Razem z
polem elektrycznym wzdłuż osi nadaje elektronom charakterystyczny dla silnika
Halla ruch w poprzek, a jednocześnie zatrzymuje je w kanale. W obszarze silnego
pola elektrony, które rzadko się zderzają, rozpędzają się szybciej, niż tracą
energię, i formują wiązkę widoczną jako osobna smuga w rozkładzie prędkości oraz
jako ogon w rozkładzie energii.

## Układ plików

```
run.py                 uruchamianie wersji jednowymiarowej
run2d.py               uruchamianie wersji na płaszczyźnie
make_figures.py        gotowe rysunki podsumowujące
make_density_maps.py   mapy gęstości z wiązką i warstwami przyściennymi
requirements.txt

hall_pic/              wersja jednowymiarowa
  config.py            wszystkie nastawy
  constants.py         stałe fizyczne
  cross_sections.py    prawdopodobieństwa zderzeń dla ksenonu
  species.py           pojemnik na czastki modelowe
  poisson.py           potencjał i pole elektryczne
  circuit.py           zewnętrzny obwód zasilający
  pusher.py            rozkładanie ładunku, odczyt pola, ruch czastek, brzegi
  collisions.py        zderzenia z gazem
  apr.py               dzielenie i łączenie czastek
  diagnostics.py       podgląd na żywo
  simulation.py        główna pętla

hall_pic2d/            wersja na płaszczyźnie
  config2d.py          nastawy z wyborem płaszczyzny kanału
  poisson2d.py         potencjał i pole na płaszczyźnie
  species2d.py         pojemnik na czastki modelowe
  pusher2d.py          operacje siatka-czastka i ruch czastek
  collisions2d.py      zderzenia z gazem
  apr2d.py             dzielenie i łączenie czastek
  diagnostics2d.py     podgląd na żywo z mapami
  simulation2d.py      główna pętla
```

Moduły stałych fizycznych, prawdopodobieństw zderzeń i obwodu są wspólne dla
obu wersji, bo ta fizyka nie zależy od liczby wymiarów.

## Dwie płaszczyzny w wersji dwuwymiarowej

Przy geometrii `z-theta` liczymy na płaszczyźnie obejmującej oś kanału i
kierunek wokół niego. Ten drugi kierunek zawija się sam w sobie, bo biegnie po
okręgu, a pole magnetyczne wychodzi wtedy prostopadle poza płaszczyznę. Taka
płaszczyzna dobrze pokazuje niestabilność ruchu elektronów wokół kanału.

Przy geometrii `z-r` liczymy na płaszczyźnie obejmującej oś kanału i kierunek w
poprzek, od jednej ścianki do drugiej. Ścianki pochłaniają czastki, a pole
magnetyczne leży wtedy w płaszczyźnie. Ta płaszczyzna pokazuje straty na
ściankach i warstwy przy nich.

## Świadome uproszczenia

To model badawczy, a nie narzędzie inżynierskie, więc kilka rzeczy jest
uproszczonych celowo:

- Prawdopodobieństwa zderzeń dla ksenonu to gładkie przybliżenia, dobrane tak,
  by tempo jonizacji i strat energii zgadzało się z danymi odniesienia oraz z
  pomiarami. Do zastosowań ilościowych warto podstawić pełne dane tablicowe.
- Gaz roboczy jest zamrożony, to znaczy jego rozkład jest z góry ustalony i nie
  zmienia się w czasie.
- Sprzężenie z obwodem opiera się na ładunku zebranym na anodzie. Do pełnej
  zgodności bilansu ładunku przydałby się dodatkowy człon prądu przesunięcia.
- W wersji `z-r` ścianki są uziemione. Prawdziwa ceramika pływa blisko
  potencjału plazmy, gromadzi ładunek i emituje elektrony wtórne, co tu jest
  pominięte.
- Model katody jest prosty: dosyła elektrony w liczbie powiązanej ze strumieniem
  jonów, bez pełnej fizyki katody.

## Powtarzalność

Każdy przebieg startuje z ustalonym ziarnem losowości, więc przy tych samych
nastawach daje ten sam wynik. Ziarno można zmienić opcją `--seed`.
