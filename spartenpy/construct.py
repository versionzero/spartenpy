
import numpy as np
from scipy.sparse import rand
from sktensor import sptensor

import spartenpy as sp
from .coo import *

def fromarray(A):
    """Create a sptensor from a dense numpy array"""
    subs = np.nonzero(A)
    vals = A[subs]
    return sptensor(subs, vals, shape=A.shape, dtype=A.dtype)


def testten():
    a =  np.array([[(0,  1, 0,  2),
           [3,  0, 4,  0],
           [0,  5, 6,  0]],
          [[ 7, 0, 8,  0],
           [ 0, 9, 10, 0],
           [11, 0, 0,  12]]])
    st = fromarray(a)
    return sp.coo_tensor(st)


def rand(L, M, N):
    a = np.random.randint(2, size=(L, M, N))
    st = fromarray(a)
    return sp.coo_tensor(st)
