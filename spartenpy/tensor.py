""" Sparse tensor representations """

import numpy as np
from sktensor import sptensor

def cumsumdups(Ai, size):
    """
    Calculate the CUMulative SUM of duplicates. NOTE: this assumes
    duplicates are contiguous.
    """
    
    counts = [0] * size
    for i in Ai:
        counts[i+1] += 1
    return np.cumsum(counts).tolist()


def coo_tocsr(n_row, n_col, Ai, Aj, Ak, Ax):
    """
    Compute B = A for COO matrix A, CSR matrix B
    
    Input Arguments:
      I  n_row      - number of rows in A
      I  n_col      - number of columns in A
      I  nnz        - number of nonzeros in A
      I  Ai[nnz(A)] - row indices
      I  Aj[nnz(A)] - column indices
      T  Ax[nnz(A)] - nonzeros
    Output Arguments:
      I Bp  - row pointer
      I Bj  - column indices
      T Bx  - nonzeros

    Note:
      Output arrays Bp, Bj, and Bx must be preallocated

    Note:
      Input:  row and column indices *are not* assumed to be ordered
    
    Note:
      Duplicate entries are carried over to the CSR represention

      Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))

    Note:
      Adapted from scipy.sparse.
    """

    nnz = len(Ax)

    Bp = cumsumdups(Ai, n_row)
    Bj = [0] * nnz
    Bk = [0] * nnz
    Bx = [0] * nnz

    for n in range(nnz):

        row = Ai[n]
        dest = Bp[row]

        Bj[dest] = Aj[n]
        Bk[dest] = Ak[n]
        Bx[dest] = Ax[n]

        Bp[row] += 1

    Bp = [0] + [Bp[i] for i in range(n_row-1)]
    
    return Bp, Bj, Bk, Bx


def crs_toccs(n_row, n_col, Ap, Aj, Ak, Ax):
    """
    Compute B = A for CSR matrix A, CSC matrix B
    
    Also, with the appropriate arguments can also be used to:
      - compute B = A^t for CSR matrix A, CSR matrix B
      - compute B = A^t for CSC matrix A, CSC matrix B
      - convert CSC->CSR
    
    Input Arguments:
      I  n_row         - number of rows in A
      I  n_col         - number of columns in A
      I  Ap[n_row+1]   - row pointer
      I  Aj[nnz(A)]    - column indices
      T  Ax[nnz(A)]    - nonzeros
    Output Arguments:
      I  Bp[n_col+1] - column pointer
      I  Bj[nnz(A)]  - row indices
      T  Bx[nnz(A)]  - nonzeros
    
    Note:
      Output arrays Bp, Bj, Bx must be preallocated
    
    Note:
      Input:  column indices *are not* assumed to be in sorted order
      Output: row indices *will be* in sorted order

      Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))

    Note:
      Adapted from scipy.sparse.
    """
    nnz = Ap[n_row]

    Bp = cumsumdups(Aj, n_col+1)
    Bi = [0] * nnz
    Bk = [0] * nnz
    Bx = [0] * nnz

    for row in range(n_row):
        for jj in range(Ap[row], Ap[row+1]):
           
            col = Aj[jj]
            dest = Bp[col]

            print (Ak[jj], Ax[jj])

            Bi[dest] = row
            Bk[dest] = Ak[jj]
            Bx[dest] = Ax[jj]

            Bp[col] += 1

    Bp = [0] + [Bp[i] for i in range(n_col-1)]
    
    return Bp, Bi, Bk, Bx
        
class coo_tensor(object):

    def __init__(self, tensor):
        
        if not isinstance(tensor, sptensor):
            raise TypeError("Must be a sptensor")

        self.tensor = tensor

    def tocsr(self):
        tensor = self.tensor
        L, M, N = tensor.shape
        subs = tensor.subs
        Aj = subs[2]
        Ai = subs[1]
        Ak = subs[0]
        Ax = tensor.vals
        Bp, Bj, Bk, Bx = coo_tocsr(M+1, N, Ai, Aj, Ak, Ax)
        return crs_tensor(Bp, Bj, Bk, Bx,
                          shape=tensor.shape,
                          dtype=tensor.dtype)

    def toccs(self):
        tensor = self.tensor
        L, M, N = tensor.shape
        subs = tensor.subs
        Aj = subs[2]
        Ai = subs[1]
        Ak = subs[0]
        Ax = tensor.vals
        Bp, Bj, Bk, Bx = coo_tocsr(N+1, M, Aj, Ai, Ak, Ax)
        return ccs_tensor(Bp, Bj, Bk, Bx,
                          shape=tensor.shape,
                          dtype=tensor.dtype)
        
    def __repr__(self):
        return repr(self.tensor)

    def __str__(self):
        return self.__repr__()


class ccs_tensor(object):

    def __init__(self, RO, CO, KO, vals, shape=None, dtype=None):
        self.nnz = len(vals)
        self.RO = RO
        self.CO = CO
        self.KO = KO
        self.vals = vals
        self.shape = shape
        self.dtype = dtype


class crs_tensor(object):

    def __init__(self, RO, CO, KO, vals, shape=None, dtype=None):
        self.nnz = len(vals)
        self.RO = RO
        self.CO = CO
        self.KO = KO
        self.vals = vals
        self.shape = shape
        self.dtype = dtype

    def toccs(self):
        L, M, N = self.shape
        Ap = self.RO
        Aj = self.CO
        Ak = self.KO
        Ax = self.vals
        Bp, Bi, Bk, Bx = crs_toccs(M, N+1, Ap, Aj, Ak, Ax)
        return ccs_tensor(Bp, Bi, Bk, Bx,
                          shape=self.shape,
                          dtype=self.dtype)

