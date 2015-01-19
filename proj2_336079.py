import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from itertools import product
from collections import deque


kto = 'Tomasz Zakrzewski'


def fraktal(p, x_min, x_max, y_min, y_max, delta, a=1, n=30, c=5, eps=1.0e-15):
    rozm_n = np.int((x_max - x_min) / delta)
    rozm_m = np.int((y_max - y_min) / delta)
    q = deque()
    zera = p.roots()
    pochodna = p.deriv(1)
    punkty = np.fromfunction(lambda k, l: (x_min + l * delta) + 1j * (y_max - k * delta), (rozm_m, rozm_n))
    iteracja_koniec = np.zeros((rozm_m, rozm_n),dtype = np.int)
    pierwiastek = np.zeros((rozm_m, rozm_n),dtype = np.complex)
    q.append(punkty.copy())
    for i in range(n):
        punkty = punkty - a * p(punkty) / pochodna(punkty)
        for k, l in product(range(rozm_m), range(rozm_n)):
            if iteracja_koniec[k, l] == 0:
                pkt = punkty[k, l]
                id_cyklu = i - len(q) + 1
                for poprz in q:
                    if abs(pkt - poprz[k, l]) <= eps:
                        pierwiastek[k, l] = (pkt + poprz[k, l]) / 2
                        iteracja_koniec[k, l] = id_cyklu
                    id_cyklu += 1

        if len(q) + 1 > c:
            q.popleft()
        q.append(punkty.copy())
    mapa = np.zeros((rozm_m, rozm_n), dtype=np.int)
    for k, l in product(range(rozm_m), range(rozm_n)):
        mapa[k,l] = pierwiastek[k, l] + n * iteracja_koniec[k, l]
    return mapa, punkty


w = Polynomial([-1, 0, 0, 1]) #z^3 - 1
f, wyniki = fraktal(w, -1, 1, -1, 1, 0.003, n=10, c=7)
plt.figure(figsize=(8, 8))
plt.imshow(f)