import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates
from numba import njit

from scipy import sparse




# @njit
# this is slow AF
# def rvnumpy(daydates:np.ndarray, closingprices:np.ndarray)->np.ndarray:
#     logreturnsquared = (np.log(closingprices[1:])-np.log(closingprices[:-1]))**2
#     returndates = np.unique(daydates[1:])
#     # dailyrealizedvariance = np.zeros((len(returndates),))
#     # for index, thatday in enumerate(returndates):
#     #     dailyrealizedvariance[index] = np.sum(logreturnsquared[daydates[1:]==thatday)
#     dailyrealizedvariance = np.array([(np.sum(logreturnsquared[daydates[1:]==thatday])) for thatday in returndates])
#     return dailyrealizedvariance


# This won't work until dates are converted to int
# def rvscipy(daydates:np.ndarray, closingprices:np.ndarray)->np.ndarray:
#     logreturnsquared = np.array((np.log(closingprices[1:])-np.log(closingprices[:-1]))**2)
#     returndates, ids = np.unique(daydates[1:], return_inverse=True)

#     # https://stackoverflow.com/a/49143979
#     x_sum = logreturnsquared.sum(axis=0)
#     groups = daydates[1:]

#     c = np.array(sparse.csr_matrix(
#         (
#             x_sum,
#             groups,
#             np.arange(len(groups)+1)
#         ),
#         shape=(len(groups), len(returndates))
#     ).sum(axis=0)).ravel()

#     return c

















"""
# @njit
# Exception has occurred: TypingError
# Failed in nopython mode pipeline (step: nopython frontend)
# [1m[1mUse of unsupported NumPy function 'numpy.insert' or unsupported use of the function.
def _running_mean(x, N):
    # https://stackoverflow.com/a/27681394 
    # I added my own twist to keep the variables the same length
    cumsum = np.cumsum(np.insert(x, 0, 0))
    # cumsum = np.zeros((len(x)+1,))
    # cumsum = np.cumsum(cumsum)
    movavg = (cumsum[N:] - cumsum[:-N]) / float(N)
    padding = cumsum[1:N] / np.arange(N)[1:N]
    return np.insert(movavg, 0, padding)
    # cumsum = (cumsum[N:] - cumsum[:-N]) / float(N)
    # padding = cumsum[1:N] / np.arange(N)[1:N]
    # return np.insert(movavg, 0, padding)
"""