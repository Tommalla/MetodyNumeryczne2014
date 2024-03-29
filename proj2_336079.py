import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from IPython.display import display, Math


kto = 'Tomasz Zakrzewski'


def rowne(x, y, eps):
    return abs(x - y) <= eps


def fraktal(p, x_min, x_max, y_min, y_max, delta, a=1, n=30, c=5, eps=1.0e-15):
    """ Klasyfkuje punkty startowe dla uogolnionej metody Newtona w zależności
    od granicznego zachowania ciągu iteracji.

    Args:
        p - wielomian
        x_min, x_max, y_min, y_max - graniczne współrzędne dla prostokąta punktów startowych,
        delta - odległość w pionie i poziomie pomiędzy sąsiednimi punktami startowymi.
        a - parametr modyfikujący metodę Newtona,
        n - ilość iteracji
        c - ograniczenie górne na długość rozpoznawanego cyklu
        eps - dokłaność stosowana do rozpoznawania granic ciągu jako pierwiastków i/lub cykliczności

    Funkcja zwraca tablicę liczbową (int lub float) m, w której zakodowana jest klasyfikacja ciągu iteracji:
    dla każdego punktu startowego: m[k, l] określa zachowanie punktu x_min + l * delta + 1j * y_max - k * delta
    """
    max_pierw = len(p.roots())
    rozm_n = np.int((x_max - x_min) / delta)
    rozm_m = np.int((y_max - y_min) / delta)
    punkty = np.fromfunction(lambda k, l: (x_min + l * delta) + 1j * (y_max - k * delta), (rozm_m, rozm_n))
    punkty = punkty.flatten()
    kol = deque()
    kol.append(punkty.copy())
    pochodna = p.deriv(1)
    ostatnia_iteracja = np.zeros(shape=punkty.shape, dtype=int)  # Numer iteracji po której osiągnęliśmy zbieżność/cykliczność.
    niezbiezne = np.ones(shape=punkty.shape, dtype=bool)  # Prawda dla każdego punktu, który nie jest jeszcze zbieżny.
    jest_cykl = np.zeros(shape=punkty.shape, dtype=bool)
    wynik = np.zeros(shape=punkty.shape, dtype=int)
    indeksy = np.arange(len(punkty))
    for i in range(n):
        wartosci = p(punkty[niezbiezne])
        nowe_niezbiezne = np.abs(wartosci) > eps
        ostatnia_iteracja[indeksy[niezbiezne][~nowe_niezbiezne]] = i
        niezbiezne[niezbiezne] = nowe_niezbiezne
        punkty[niezbiezne] -= a * wartosci[nowe_niezbiezne] / pochodna(punkty[niezbiezne])

        if len(kol) + 1 > c:
            kol.popleft()
        kol.append(punkty.copy())

    for i in range(2):  # Pozbywamy się cykli dł. 0 i 1
        if len(kol) > 0:
            kol.pop()
    for k in range(len(kol)):
        poprz = kol[k]
        for j in range(len(poprz)):
            if niezbiezne[j] and rowne(punkty[j], poprz[j], eps):
                jest_cykl[j] = 1
                niezbiezne[j] = 0

    ostatni_nr = 1
    s = set()
    for i in range(len(punkty)):
        if not niezbiezne[i]:
            obecny = False
            numer = 0
            for pkt, nr in s:
                if rowne(pkt, punkty[i], eps):
                    obecny = True
                    numer = nr
                    break

            if not obecny:
                s.add((punkty[i], ostatni_nr + 1))
                ostatni_nr += 1
                numer = ostatni_nr

            if jest_cykl[i]:
                wynik[i] = (numer + max_pierw) * n
            else:
                wynik[i] = ostatnia_iteracja[i] + numer * n

    return wynik.reshape((rozm_m, rozm_n))


def poly_2_latex(coefs):
    s = '{:.2f}'.format(coefs[0]) if coefs[0] else ''
    for c, k in zip(coefs[1:], range(1, len(coefs))):
        if c > 0 and s:
            s += '+ {:.2f}x^{{ {} }}'.format(c, k)
        elif c < 0 or not s:
            s += ' {:.2f}x^{{ {} }}'.format(c, k)
    return Math(s)


def demo():
    plt.ion()
    print('Każdy wielomian po 30 iteracji, każdy na przedziale -1 - i do 1 + i, delta = 0.003')
    dane = [
        ([-1, 0, 0, 1], 1.0, 0, 1e-15), # Wielomian, a, c
        ([-1, 0, 0, 1], 1.0 + 0.1j, 5, 1e-15),
        ([-16, 0, 0, 0, 15, 0, 0, 0, 1], 1.0, 7, 1e-10),
        ([-16, 0, 0, 0, 15, 0, 0, 0, 1], 1.1 + 0.2j, 14, 1e-10),
        ([-1, 0, 0, 1, 0, 0, 1], 1.0, 7, 1e-9),
    ]
    for lista, a, c, eps in dane:
        w = Polynomial(lista)
        pocz = time.time()
        wynik = fraktal(w, -1, 1, -1, 1, 0.003, a=a, c=c, eps=eps)
        display(poly_2_latex(lista))
        print(' a =', a, ' c =', c, 'eps =', eps, 'czas:', time.time() - pocz)
        plt.figure(figsize=(8, 8))
        plt.imshow(wynik)
        plt.show()