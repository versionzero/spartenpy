
import numpy as np
from scipy.sparse import rand
from sktensor import sptensor

import spartenpy as sp
from .tensor import *


def rand(L, M, N, orient=sp.array.frontal):
    a = np.random.randint(2, size=(L, M, N))
    a = orient(a)
    st = sp.tensor.fromarray(a, orient)
    return sp.coo_tensor(st)
