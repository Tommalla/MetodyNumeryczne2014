# -*- coding: utf-8 -*-
from scipy import sparse
import numpy as np


kto = 'Tomasz Zakrzewski'


def swap_rows(A, a, b):
    """Swapuje wiersze a i b macierzy A.

    Założenie: Macierz A jest obiektem klasy scipy.sparse.csc_matrix.
    """
    a_idx = np.where(A.indices == a)
    b_idx = np.where(A.indices == b)
    A.indices[a_idx] = b
    A.indices[b_idx] = a
    return


def tridiag_lu(A):
    """Wyznacza rozkład PA=LU macierzy trójdiagonalnej A.
    A jest obiektem klasy scipy.sparse.dia_matrix.

    Zwraca macierze P, L, U, przy czym:
    - L, U to obiekty klasy scipy.sparse.dia_matrix
    - P to obiekt klasy scipy.sparse.dok_matrix
    """
    A = sparse.csc_matrix(A)
    shape = A.get_shape()
    n, _ = shape
    L = sparse.csc_matrix(sparse.identity(n))
    P = sparse.csc_matrix(shape)
    r = [i for i in range(0, n)]
    ops = 0

    for p in range(0, n - 1):
        for j in range(p + 1, n):
            if abs(A[r[j], p]) > abs(A[r[p], p]):
                # Zamieniamy wiersze
                tmp = r[p]
                r[p] = r[j]
                r[j] = tmp

        for k in range(p + 1, n):
            A[r[k], p] = A[r[k], p] / A[r[p], p]
            for c in range(p + 1, n):
                A[r[k], c] = A[r[k], c] - A[r[k], p] * A[r[p], c]
                ops += 1

    U = sparse.csc_matrix(shape)
    for i in range(0, n):
        U[i] = A[r[i]]

    for i in range(0, n):
        P[i, r[i]] = 1
        for j in range(0, i):
            L[i, j] = A[r[i], j]
            U[i, j] = 0

    print 'Done, needs converting, ops: ', ops

    P = sparse.dok_matrix(P)
    L = sparse.dia_matrix(L)
    U = sparse.dia_matrix(U)
    return P, L, U


def tridiag_solve(A, b):
    """Wyznacza rozwiązanie x układu równań Ax = b dla
    macierzy trójdiagonalej A i wektora b.
    A jest obiektem klasy scipy.sparse.dia_matrix,
    b jest 1-wymiarową tablicą (numpy.ndarray).

    Zwraca 1-wymiarową tablicę x
    """
    # ...
    # return x
    # zaalokuj x, oblicz i zwróć
    pass


def test_tridiag_lu(dl, d, du, tol=None):
    if tol is None: tol = np.finfo(np.float_).eps
    A_data = np.array([d, du, dl], dtype=np.float)
    A_offsets = np.array([0, 1, -1])
    n = len(d)
    A = sparse.dia_matrix((A_data, A_offsets), shape=(n, n), dtype=np.float_)
    P, L, U = tridiag_lu(A)
    # print 'P\n', P.todense(), '\nL\n', L.todense(), '\nU\n', U.todense()
    # print P.todense() * A.todense()
    # print (L*U).todense()
    return np.allclose(P.todense() * A.todense(), (L * U).todense(), rtol=tol, atol=0.0)

#testy (można określić "przewidywaną" tolerancję):
test_tridiag_lu([10,20,30,40,50], [2,4,6,8,10], [3,9,12,15,18])
test_tridiag_lu([1,2,3,4], [0, 0, 0, 0], [0, 0, 0, 0])
test_tridiag_lu([0.01]*100, [1.0]*100, [1.0]*100, tol=0.001)


# def test_tridiag_solve(dl, d, du, b, tol=None):
#     if tol is None: tol = np.finfo(np.float_).eps
#     A_data = np.array([d, du, dl], dtype=np.float)
#     A_offsets = np.array([0, 1, -1])
#     n = len(d)
#     A = sparse.dia_matrix((A_data, A_offsets), shape=(n, n), dtype=np.float_)
#     x = tridiag_solve(A, b)
#     return np.allclose(A.todense() * np.matrix(x).T, np.matrix(b).T, dtype=np.float_, rtol=tol, atol=0.0)
#
# #testy
# test_tridiag_solve([10,20,30,40,50], [2,4,6,8,10], [3,9,12,15,18], np.array([1, -2, 4, -8, 16]))
# test_tridiag_solve([0.01]*100, [1.0]*100, [1.0]*100, np.array([1.0]*100))