
import numpy as np


def cumsumdups(Ai, size):
    """
    Calculate the CUMulative SUM of duplicates. NOTE: this assumes
    duplicates are contiguous.
    """
    
    counts = [0] * size
    for i in Ai:
        counts[i+1] += 1
    return np.cumsum(counts).tolist()
