# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
import time


kto = 'Tomasz Zakrzewski'


def tridiag_lu(A):
    """Wyznacza rozkład PA=LU macierzy trójdiagonalnej A.
    A jest obiektem klasy scipy.sparse.dia_matrix.

    Zwraca macierze P, L, U, przy czym:
    - L, U to obiekty klasy scipy.sparse.dia_matrix
    - P to obiekt klasy scipy.sparse.dok_matrix
    """
    shape = A.get_shape()
    n, _ = shape
    A = A.todense()
    L = np.identity(n)
    P = sparse.dok_matrix(shape)
    U = np.zeros(shape=shape)
    r = range(0, n)

    for p in range(0, n - 1):
        for j in range(p + 1, p + 2):
            if abs(A[r[j], p]) > abs(A[r[p], p]):
                # Zamieniamy wiersze
                tmp = r[p]
                r[p] = r[j]
                r[j] = tmp

        for k in range(p + 1, min(p + 3, n)):
            A[r[k], p] = A[r[k], p] / A[r[p], p]
            for c in range(p + 1, min(p + 3, n)):
                A[r[k], c] = A[r[k], c] - A[r[k], p] * A[r[p], c]

    for i in range(0, n):
        U[i] = A[r[i]]

    for i in range(0, n):
        P[i, r[i]] = 1
        for j in range(0, i):
            L[i, j] = A[r[i], j]
            U[i, j] = 0

    return P, sparse.dia_matrix(L), sparse.dia_matrix(U)


def tridiag_solve(A, b):
    """Wyznacza rozwiązanie x układu równań Ax = b dla
    macierzy trójdiagonalej A i wektora b.
    A jest obiektem klasy scipy.sparse.dia_matrix,
    b jest 1-wymiarową tablicą (numpy.ndarray).

    Zwraca 1-wymiarową tablicę x
    """
    P, L, U = tridiag_lu(A)
    n = A.shape[0]
    r = range(0, n)
    for (i, pi), _ in P.iteritems():
        r[i] = pi

    L = L.todense()
    U = U.todense()
    x = [0 for _ in range(0, n)]
    y = [0 for _ in range(0, n)]

    for k in range(0, n):
        sum = 0
        for i in range(0, k):
            sum += y[i] * L[k, i]
        y[k] = (b[r[k]] - sum) / L[k, k]

    for k in reversed(range(0, n)):
        sum = 0
        for i in range(k + 1, n):
            sum += x[i] * U[k, i]
        x[k] = (y[k] - sum) / U[k, k]

    return x


def test_tridiag_lu(dl, d, du, tol=None):
    if tol is None:
        tol = np.finfo(np.float_).eps
    A_data = np.array([d, du, dl], dtype=np.float)
    A_offsets = np.array([0, 1, -1])
    n = len(d)
    A = sparse.dia_matrix((A_data, A_offsets), shape=(n, n), dtype=np.float_)
    P, L, U = tridiag_lu(A)
    return np.allclose(P.todense() * A.todense(), (L * U).todense(), rtol=tol, atol=0.0)


def test_tridiag_solve(dl, d, du, b, tol=None):
    if tol is None:
        tol = np.finfo(np.float_).eps
    A_data = np.array([d, du, dl], dtype=np.float)
    A_offsets = np.array([0, 1, -1])
    n = len(d)
    A = sparse.dia_matrix((A_data, A_offsets), shape=(n, n), dtype=np.float_)
    x = tridiag_solve(A, b)
    return np.allclose(A.todense() * np.matrix(x).T, np.matrix(b).T, rtol=tol, atol=0.0)


def testy():
    tridiag_lu_data = [
            ([10,20,30,40,50], [2,4,6,8,10], [3,9,12,15,18]),
            ([1,2,3,4], [0, 0, 0, 0], [0, 0, 0, 0]),
            ([0.01]*100, [1.0]*100, [1.0]*100),
    ]

    tridiag_solve_data = [
            ([10,20,30,40,50], [2,4,6,8,10], [3,9,12,15,18], np.array([1, -2, 4, -8, 16])),
            ([0.01]*100, [1.0]*100, [1.0]*100, np.array([1.0]*100)),
    ]

    print('Testuję tridiag_lu z minimalną tolerancją...')
    for data in tridiag_lu_data:
        dl, d, du = data
        begin = time.time()
        assert(test_tridiag_lu(dl, d, du))
        print("> Sukces! Czas wykonania testu: %f" % (time.time() - begin))

    tol = 1e-15
    print('Testuję tridiag_solve z tolerancją %e' % tol)
    for data in tridiag_solve_data:
        dl, d, du, b = data
        begin = time.time()
        assert(test_tridiag_solve(dl, d, du, b, tol=tol))
        print("> Sukces! Czas wykonywania testu: %f" % (time.time() - begin))


testy()