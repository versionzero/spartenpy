

def random(l, m, n, density=0.01, format='coo', random_state=None):
    """Generate a sparse matrix of the given shape and density with randomly
    distributed values.

    Parameters
    ----------
    l, m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {numpy.random.RandomState, int}, optional
        Random number generator or random seed. If not given, the singleton
        numpy.random will be used.  This random state will be used
        for sampling the sparsity structure, but not necessarily for sampling
        the values of the structurally nonzero entries of the matrix.
    data_rvs : callable, optional
        Samples a requested number of random values.
        This function should take a single argument specifying the length
        of the ndarray that it will return.  The structurally nonzero entries
        of the sparse random matrix will be taken from the array sampled
        by this function.  By default, uniform [0, 1) random values will be
        sampled using the same random state as is used for sampling
        the sparsity structure.

    Examples
    --------
    >>> from scipy.sparse import random
    >>> from scipy import stats
    >>> class CustomRandomState(object):
    ...     def randint(self, k):
    ...         i = np.random.randint(k)
    ...         return i - i % 2
    >>> rs = CustomRandomState()
    >>> rvs = stats.poisson(25, loc=10).rvs
    >>> S = random(3, 4, density=0.25, random_state=rs, data_rvs=rvs)
    >>> S.A
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])

    Notes
    -----
    Only float types are supported for now.
    """
    if density < 0 or density > 1:
        raise ValueError("density expected to be 0 <= density <= 1")

    lmn = l * m * n
    dtype = int
    tp = np.intc

    # Number of non zero values
    nnz = int(density * l * m * n)

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, (int, np.integer)):
        random_state = np.random.RandomState(random_state)
    if data_rvs is None:
        data_rvs = np.ones

    # Use the algorithm from python's random.sample for nnz < lmn/3.
    if lmn < 3*nnz:
        # We should use this line, but choice is only available in numpy >= 1.7
        # ind = random_state.choice(lmn, size=nnz, replace=False)
        ind = random_state.permutation(lmn)[:nnz]
    else:
        ind = np.empty(nnz, dtype=tp)
        selected = set()
        for i in xrange(nnz):
            j = random_state.randint(lmn)
            while j in selected:
                j = random_state.randint(lmn)
            selected.add(j)
            ind[i] = j

    j = np.floor(ind * 1. / m).astype(tp)

    print j
    
    i = (ind - j * m).astype(tp)

    print i
    
    k = np.ones(m).astype(tp)

    print k
    
    vals = np.ones(nnz).astype(dtype)
    #return coo_tensor((vals, (i, j, k)), shape=(l, m, n)).asformat(format)


def rand(l, m, n, density=0.01, format="coo", random_state=None):
    """Generate a sparse matrix of the given shape and density with uniformly
    distributed values.

    Parameters
    ----------
    l, m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {numpy.random.RandomState, int}, optional
        Random number generator or random seed. If not given, the singleton
        numpy.random will be used.

    Notes
    -----
    Only float types are supported for now.

    """
    return random(l, m, n, density, format, dtype, random_state)
