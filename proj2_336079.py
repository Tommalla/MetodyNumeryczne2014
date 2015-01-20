from numpy.polynomial.polynomial import Polynomial
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time


kto = 'Tomasz Zakrzewski'


def rowne(x, y, eps):
    return abs(x - y) <= eps


def fraktal(p, x_min, x_max, y_min, y_max, delta, a=1, n=30, c=5, eps=1.0e-15):
    rozm_n = np.int((x_max - x_min) / delta)
    rozm_m = np.int((y_max - y_min) / delta)
    punkty = np.fromfunction(lambda k, l: (x_min + l * delta) + 1j * (y_max - k * delta), (rozm_m, rozm_n))
    punkty = punkty.flatten()
    kol = deque()
    kol.append(punkty.copy())
    pochodna = p.deriv(1)
    ostatnia_iteracja = np.zeros(shape=punkty.shape)  # Numer iteracji po której osiągnęliśmy zbieżność/cykliczność.
    niezbiezne = np.ones(shape=punkty.shape, dtype=bool)  # Prawda dla każdego punktu, który nie jest jeszcze zbieżny.
    wynik = np.zeros(shape=punkty.shape, dtype=int)
    indeksy = np.arange(len(punkty))
    for i in range(n):
        wartosci = p(punkty[niezbiezne])
        nowe_niezbiezne = np.abs(wartosci) > eps
        ostatnia_iteracja[indeksy[niezbiezne][~nowe_niezbiezne]] = i
        niezbiezne[niezbiezne] = nowe_niezbiezne
        punkty[niezbiezne] -= a * wartosci[nowe_niezbiezne] / pochodna(punkty[niezbiezne])

        id_cyklu = i - len(kol) + 1
        for poprz in kol:
            cykle = rowne(punkty, poprz, eps)[niezbiezne]
            ostatnia_iteracja[indeksy[niezbiezne][cykle]] = id_cyklu
            niezbiezne[niezbiezne][cykle] = False
            id_cyklu = id_cyklu + 1
            # TODO średnia?

        if len(kol) + 1 > c:
            kol.popleft()
        kol.append(punkty.copy())

    ostatni_nr = 1
    s = set()
    zbiezne = punkty[~niezbiezne]
    zbiezne_indeksy = indeksy[~niezbiezne]
    for i in range(len(zbiezne)):
        obecny = False
        numer = 0
        for pkt, nr in s:
            if rowne(pkt, zbiezne[i], eps):
                obecny = True
                numer = nr
                break

        if not obecny:
            s.add((zbiezne[i], ostatni_nr + 1))
            ostatni_nr += 1
            numer = ostatni_nr

        wynik[zbiezne_indeksy[i]] = ostatnia_iteracja[zbiezne_indeksy[i]] + numer * n

    return wynik.reshape((rozm_m, rozm_n))


w = Polynomial([-1, 0, 0, 1]) #z^3 - 1
begin = time.time()
pic = fraktal(w, -1, 1, -1, 1, 0.003, a=1.0 + 0.1j, n=30)
print(time.time() - begin)
plt.figure(figsize=(8, 8))
plt.imshow(pic)
plt.show()