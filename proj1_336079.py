from scipy import sparse
import numpy as np

kto = 'Tomasz Zakrzewski'

def tridiag_lu(A):
    """Wyznacza rozkład PA=LU macierzy trójdiagonalnej A.
    A jest obiektem klasy scipy.sparse.dia_matrix.

    Zwraca macierze P, L, U, przy czym:
    - L, U to obiekty klasy scipy.sparse.dia_matrix
    - P to obiekt klasy scipy.sparse.dok_matrix
    """
    # ...
    # return P, L, U
    pass


def tridiag_solve(A, b):
    """Wyznacza rozwiązanie x układu równań Ax = b dla
    macierzy trójdiagonalej A i wektora b.
    A jest obiektem klasy scipy.sparse.dia_matrix,
    b jest 1-wymiarową tablicą (numpy.ndarray).

    Zwraca 1-wymiarową tablicę x
    """
    # ...
    # return x
    pass


def test_tridiag_lu(dl, d, du, tol=None):
    if tol is None: tol = np.finfo(np.float_).eps
    A_data = np.array([d, du, dl], dtype=np.float)
    A_offsets = np.array([0, 1, -1])
    n = len(d)
    A = sparse.dia_matrix((A_data, A_offsets), shape=(n, n), dtype=np.float_)
    P, L, U = tridiag_lu(A)
    return np.allclose(P.todense() * A.todense(), (L * U).todense(), rtol=tol, atol=0.0)

#testy (można określić "przewidywaną" tolerancję):
test_tridiag_lu([10,20,30,40,50], [2,4,6,8,10], [3,9,12,15,18], tol=...)
test_tridiag_lu([1,2,3,4], [0, 0, 0, 0], [0, 0, 0, 0], tol=...)
test_tridiag_lu([0.01]*100, [1.0]*100, [1.0]*100, tol=...)


def test_tridiag_solve(dl, d, du, b, tol=None):
    if tol is None: tol = np.finfo(np.float_).eps
    A_data = np.array([d, du, dl], dtype=np.float)
    A_offsets = np.array([0, 1, -1])
    n = len(d)
    A = sparse.dia_matrix((A_data, A_offsets), shape=(n, n), dtype=np.float_)
    x = tridiag_solve(A, b)
    return np.allclose(A.todense() * np.matrix(x).T, np.matrix(b).T, dtype=np.float_, rtol=tol, atol=0.0)

#testy
test_tridiag_solve([10,20,30,40,50], [2,4,6,8,10], [3,9,12,15,18], np.array([1, -2, 4, -8, 16]), tol=...)
test_tridiag_solve([0.01]*100, [1.0]*100, [1.0]*100, np.array([1.0]*100), tol=...)