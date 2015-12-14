""" Sparse tensor representations """

import numpy as np
from sktensor import sptensor
from utility import cumsumdups


def coo_tocrs(n_row, n_col, n_tube, Ai, Aj, Ak, Ax):
    """
    Compute B = A for COO matrix A, CRS matrix B
    
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
      Duplicate entries are carried over to the CRS represention

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


def coo_toecrs(n_row, n_col, n_tube, on, Ai, Aj, Ak, Ax):
    """
    Compute B = A for COO matrix A, CRS matrix B
    
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
      Duplicate entries are carried over to the CRS represention

      Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))

    Note:
      Adapted from scipy.sparse.
    """

    nnz = len(Ax)

    Bp = cumsumdups(Ai, n_row)
    Bjk = [0] * nnz
    Bx = [0] * nnz

    for n in range(nnz):

        row = Ai[n]
        dest = Bp[row]

        Bjk[dest] = Aj[n] * n_col + Ak[n]
        Bx[dest] = Ax[n]

        Bp[row] += 1

    Bp = [0] + [Bp[i] for i in range(n_row-1)]
    
    return Bp, Bjk, Bx


def crs_toccs(n_row, n_col, n_tube, Ap, Aj, Ak, Ax):
    """
    Compute B = A for CRS matrix A, CSC matrix B
    
    Also, with the appropriate arguments can also be used to:
      - compute B = A^t for CRS matrix A, CRS matrix B
      - compute B = A^t for CSC matrix A, CSC matrix B
      - convert CSC->CRS
    
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

            Bi[dest] = row
            Bk[dest] = Ak[jj]
            Bx[dest] = Ax[jj]

            Bp[col] += 1

    Bp = [0] + [Bp[i] for i in range(n_col-1)]
    
    return Bp, Bi, Bk, Bx


class array(object):

    @staticmethod
    def lateral(a):
        """Extract lateral slices of a 3d tensor"""
        return np.swapaxes(a, 0, 2)

    @staticmethod
    def frontal(a):
        """Extract frontal slices of a 3d tensor"""
        # It is assumed that all input arrays are in frontal format
        return a

    @staticmethod
    def vertical(a):
        """Extract vertical slices of a 3d tensor"""
        return np.swapaxes(a, 1, 2)


class tensor(object):

    @staticmethod
    def fromarray(A):
        """Create a sptensor from a dense numpy array"""
        subs = np.nonzero(A)
        vals = A[subs]
        return sptensor(subs, vals, shape=A.shape, dtype=A.dtype)


class coo_tensor(object):

    def __init__(self, tensor):
        
        if not isinstance(tensor, sptensor):
            raise TypeError("Must be a sptensor")

        self.tensor = tensor

    def tocrs(self):
        tensor = self.tensor
        L, M, N = tensor.shape
        subs = tensor.subs
        Aj = subs[2]
        Ai = subs[1]
        Ak = subs[0]
        Ax = tensor.vals
        Bp, Bj, Bk, Bx = coo_tocrs(M+1, N, L, Ai, Aj, Ak, Ax)
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
        Bp, Bj, Bk, Bx = coo_tocrs(N+1, M, L, Aj, Ai, Ak, Ax)
        return ccs_tensor(Bp, Bj, Bk, Bx,
                          shape=tensor.shape,
                          dtype=tensor.dtype)

    def tocts(self):
        tensor = self.tensor
        L, M, N = tensor.shape
        subs = tensor.subs
        Aj = subs[2]
        Ai = subs[1]
        Ak = subs[0]
        Ax = tensor.vals
        Bp, Bj, Bk, Bx = coo_tocrs(L+1, M, N, Ak, Aj, Ai, Ax)
        return cts_tensor(Bp, Bj, Bk, Bx,
                          shape=tensor.shape,
                          dtype=tensor.dtype)

    def toecrs(self):
        tensor = self.tensor
        L, M, N = tensor.shape
        subs = tensor.subs
        # These differ from the CRS subs because the orientation of
        # the input tensor is different than the original (we do this
        # to avoid sorting the input).
        Aj = subs[0]
        Ai = subs[1]
        Ak = subs[2]
        Ax = tensor.vals
        Bp, Bjk, Bx = coo_toecrs(M+1, N, L, 0, Ai, Aj, Ak, Ax)
        return ecrs_tensor(Bp, Bjk, Bx,
                           shape=tensor.shape,
                           dtype=tensor.dtype)

    def toeccs(self):
        tensor = self.tensor
        L, N, M = tensor.shape

        print (L, M, N)
        
        subs = tensor.subs
        # These differ from the CRS subs because the orientation of
        # the input tensor is different than the original (we do this
        # to avoid sorting the input).
        Aj = subs[1]
        Ai = subs[2]
        Ak = subs[0]

        print Ai
        print Aj
        print Ak
        print tensor.vals
        
        Ax = tensor.vals
        Bp, Bjk, Bx = coo_toecrs(L*N+1, N, L, 1, Aj, Ai, Ak, Ax)
        return eccs_tensor(Bp, Bjk, Bx,
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
        Bp, Bi, Bk, Bx = crs_toccs(M, N+1, L, Ap, Aj, Ak, Ax)
        return ccs_tensor(Bp, Bi, Bk, Bx,
                          shape=self.shape,
                          dtype=self.dtype)


class cts_tensor(object):

    def __init__(self, RO, CO, KO, vals, shape=None, dtype=None):
        self.nnz = len(vals)
        self.RO = RO
        self.CO = CO
        self.KO = KO
        self.vals = vals
        self.shape = shape
        self.dtype = dtype


class ecrs_tensor(object):

    def __init__(self, R, CK, vals, shape=None, dtype=None):
        self.nnz = len(vals)
        self.R = R
        self.CK = CK
        self.vals = vals
        self.shape = shape
        self.dtype = dtype


class eccs_tensor(object):

    def __init__(self, R, CK, vals, shape=None, dtype=None):
        self.nnz = len(vals)
        self.R = R
        self.CK = CK
        self.vals = vals
        self.shape = shape
        self.dtype = dtype
