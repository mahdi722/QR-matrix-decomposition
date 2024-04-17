import math
import numpy as np

def multiply_matrices(M, N):
    """
    Multiply two matrices.

    Args:
    M, N: Matrices to multiply.

    Returns:
    Result of the matrix multiplication.
    """
    tuple_N = zip(*N)
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]

def transpose_matrix(M):
    """
    Transpose a matrix.

    Args:
    M: Matrix to transpose.

    Returns:
    Transposed matrix.
    """
    n = len(M)
    return [[M[i][j] for i in range(n)] for j in range(n)]

def norm(x):
    """
    Calculate the norm of a vector.

    Args:
    x: Vector.

    Returns:
    Norm of the vector.
    """
    return math.sqrt(sum(x_i**2 for x_i in x))

def Q_i(Q_min, i, j, k):
    """
    Helper function for Householder transformation.

    Args:
    Q_min: Q matrix.
    i, j, k: Indices.

    Returns:
    Value for Q.
    """
    if i < k or j < k:
        return float(i == j)
    else:
        return Q_min[i - k][j - k]

def householder(A):
    """
    Perform Householder transformation on matrix A.

    Args:
    A: Input matrix.

    Returns:
    Q and R matrices.
    """
    n = len(A)
    R = A
    Q = [[0.0] * n for _ in range(n)]

    for k in range(n - 1):
        I = [[float(i == j) for i in range(n)] for j in range(n)]
        x = [row[k] for row in R[k:]]
        e = [row[k] for row in I[k:]]
        alpha = -math.copysign(norm(x), x[0])
        u = [p + alpha * q for p, q in zip(x, e)]
        norm_u = norm(u)
        v = [p / norm_u for p in u]

        Q_min = [[float(i == j) - 2.0 * v[i] * v[j] for i in range(n - k)] for j in range(n - k)]
        Q_t = [[Q_i(Q_min, i, j, k) for i in range(n)] for j in range(n)]

        if k == 0:
            Q = Q_t
            R = multiply_matrices(Q_t, A)
        else:
            Q = multiply_matrices(Q_t, Q)
            R = multiply_matrices(Q_t, R)

    return transpose_matrix(Q), R

def givens_rotation(A):
    """
    Perform Givens rotation on matrix A.

    Args:
    A: Input matrix.

    Returns:
    Q and R matrices.
    """
    num_rows, num_cols = np.shape(A)
    Q = np.identity(num_rows)
    R = np.copy(A)

    rows, cols = np.tril_indices(num_rows, -1, num_cols)
    for row, col in zip(rows, cols):
        if R[row, col] != 0:
            c, s = _givens_rotation_matrix_entries(R[col, col], R[row, col])

            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return Q, R

def _givens_rotation_matrix_entries(a, b):
    """
    Calculate entries for Givens rotation matrix.

    Args:
    a, b: Entries from matrix.

    Returns:
    c, s: Entries for Givens rotation matrix.
    """
    r = math.hypot(a, b)
    c = a / r
    s = -b / r

    return c, s

# Example usage:
A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
Q, R = householder(A)
