from numpy.polynomial.polynomial import Polynomial
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time


kto = 'Tomasz Zakrzewski'


def rowne(x, y, eps):
    return abs(x - y) <= eps


def newton_fractal(p, x_min, x_max, y_min, y_max, delta, a=1, n=30, c=5, eps=1.0e-15):
    rozm_n = np.int((x_max - x_min) / delta)
    rozm_m = np.int((y_max - y_min) / delta)
    arr = np.fromfunction(lambda k, l: (x_min + l * delta) + 1j * (y_max - k * delta), (rozm_m, rozm_n))
    arr = arr.flatten()
    q = deque()
    q.append(arr.copy())
    fp = p.deriv(1)
    counts = np.zeros(shape=arr.shape)
    unconverged = np.ones(shape=arr.shape, dtype=bool)
    indices = np.arange(len(arr))
    for i in range(n):
        f_g = p(arr[unconverged])
        new_unconverged = np.abs(f_g) > eps
        counts[indices[unconverged][~new_unconverged]] = i
        unconverged[unconverged] = new_unconverged
        arr[unconverged] -= a * f_g[new_unconverged] / fp(arr[unconverged])

        id_cyklu = i - len(q) + 1
        for poprz in q:
            cykle = rowne(arr, poprz, eps)[unconverged]
            counts[indices[unconverged][cykle]] = id_cyklu
            unconverged[unconverged][cykle] = False
            id_cyklu = id_cyklu + 1

        if len(q) + 1 > c:
            q.popleft()
        q.append(arr.copy())

    arr = arr.reshape((rozm_m, rozm_n))
    counts = counts.reshape((rozm_m, rozm_n))
    unconverged = unconverged.reshape((rozm_m, rozm_n))

    wynik = np.zeros(shape=counts.shape, dtype=int)

    ostatni_nr = 1
    s = set()
    for k, l in product(range(rozm_m), range(rozm_n)):
        numer = 0
        if unconverged[k, l] != 1:
            obecny = False
            numer = ostatni_nr
            for pkt, nr in s:
                if rowne(pkt, arr[k, l], eps):
                    obecny = True
                    numer = nr
                    break

            if not obecny:
                s.add((arr[k, l], ostatni_nr + 1))
                ostatni_nr += 1
                numer = ostatni_nr

        wynik[k, l] = counts[k, l] + numer * n
    return wynik


w = Polynomial([-1, 0, 0, 1]) #z^3 - 1
begin = time.time()
pic = newton_fractal(w, -1, 1, -1, 1, 0.003, a=1.0 + 0.1j, n=30)
print(time.time() - begin)
plt.figure(figsize=(8, 8))
plt.imshow(pic)
plt.show()