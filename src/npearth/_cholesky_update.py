import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def cholupdate(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Perform a Cholesky rank-one update.

    Parameters:
    L : ndarray
        Lower triangular matrix (Cholesky factor).
    x : ndarray
        Vector used for the rank-one update.

    Returns:
    L : ndarray
        Updated lower triangular matrix.
    """
    n = len(x)
    for k in range(n):
        r = np.sqrt(L[k, k] ** 2 + x[k] ** 2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k < n - 1:
            L[k + 1 : n, k] = (L[k + 1 : n, k] + s * x[k + 1 : n]) / c
            x[k + 1 : n] = c * x[k + 1 : n] - s * L[k + 1 : n, k]
    return L


@njit(cache=True, fastmath=True)
def choldowndate(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Perform a Cholesky rank-one downdate. A - x * x^T

    Parameters:
    L : ndarray
        Lower triangular matrix (Cholesky factor).
    x : ndarray
        Vector used for the rank-one downdate.

    Returns:
    L : ndarray
        Updated lower triangular matrix.
    """
    n = len(x)
    for k in range(n):
        r = np.sqrt(L[k, k] ** 2 - x[k] ** 2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k < n - 1:
            L[k + 1 : n, k] = (L[k + 1 : n, k] - s * x[k + 1 : n]) / c
            x[k + 1 : n] = c * x[k + 1 : n] - s * L[k + 1 : n, k]
    return L


@njit(cache=True, fastmath=True)
def cholsolve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system Ax = b using the Cholesky factorization of A.

    Parameters:
    L : ndarray
        Lower triangular matrix (Cholesky factor).
    b : ndarray
        Right-hand side vector.

    Returns:
    x : ndarray
        Solution vector.
    """
    n = len(b)
    # Forward substitution to solve Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    # Backward substitution to solve L^Tx = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (
            y[i]
            - np.ascontiguousarray(L[i + 1 : n, i]) @ np.ascontiguousarray(x[i + 1 : n])
        ) / L[i, i]
    return x


def _warmup_jit():
      L = np.eye(4)
      x = np.ones(4)
      cholupdate(L.copy(), x.copy())
      choldowndate(L.copy(), x.copy())
      cholsolve(L.copy(), x.copy())
      
_warmup_jit()


if __name__ == "__main__":
    # Example matrix A and its Cholesky factor L
    X = np.random.random((1000, 1000))
    A = np.matmul(X.transpose(), X)
    L = np.linalg.cholesky(A.copy())

    # Vector u for the update (rank-1 update)
    u = np.random.normal(size=A.shape[0])

    # Perform the rank-1 Cholesky update
    L_updated = cholupdate(L.copy(), u.copy())
    L_raw = np.linalg.cholesky(A.copy() + np.outer(u, u))

    print(f"max update squared error: {np.max((L_updated - L_raw) ** 2)}")

    x_np = np.linalg.solve(A, u)
    x_ch = cholsolve(L, u)
    print(f"max solve squared error: {np.max((x_np - x_ch) ** 2)}")

    assert np.all((L_updated - L_raw) ** 2 < 1e-10)
