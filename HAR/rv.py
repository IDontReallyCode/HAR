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

def rv(data:pd.DataFrame, datecolumnname='date', closingpricecolumnname='price'):
    """ 
    This function requires a dataframe with two columns ['date'] and ['price'].
    The column ['date'] needs to be just a date. No time.
    returns a tuple( numpy array of daily realized variance, numpy array of dates (text))
    """

    data['lr2'] = (np.log(data[closingpricecolumnname]) - np.log(data[closingpricecolumnname].shift(1)))**2
    data = data[data['lr2'].notna()]

    alldays = data[datecolumnname].unique()
    nbdays = len(alldays)
    realizeddailylogrange = np.zeros((nbdays,))

    idx=0
    for day, g in data.groupby(datecolumnname):
        realizeddailylogrange[idx] = sum(g['lr2'])
        if np.sqrt(realizeddailylogrange[idx]*252)<0.01:
            print(f"looks like you have an issue with data being classified as weekends. Check the time zones Example, on date: {g[datecolumnname].iloc[0]}")
        idx+=1

    return realizeddailylogrange, alldays


def rvaggregate(dailyrv: np.ndarray, aggregatesampling: list=[1,5,10,20]):
    """
    convenient function to aggregate the realized variance at various time horizon
    returns one list of numpy vectors. One vector for each time horizon
    """
    aggregated = np.zeros((len(dailyrv),len(aggregatesampling)))
    for index, sampling in enumerate(aggregatesampling):
        aggregated[:,index] = _running_meanba(dailyrv,sampling)
        # test = _running_meanba(dailyrv,sampling)
        # chek=1

    return aggregated


@njit
def _running_meanba(x, N):
    # based on https://stackoverflow.com/a/27681394 
    # but avoids using insert and keep the vector length
    # also numba possible now
    cumsum = np.zeros((len(x)+1,1))
    cumsum[1:,0] = np.cumsum(x)
    cumsum[N:,0] = (cumsum[N:,0] - cumsum[:-N,0]) / float(N)
    cumsum[1:N,0] = cumsum[1:N,0] / np.arange(N)[1:N]
    return cumsum[1:,0] 


def rq(data:pd.DataFrame):
    """ 
    This function requires a dataframe with two columns ['date'] and ['price'].
    The column ['date'] needs to be just a date. No time.
    returns a tuple( numpy array of daily realized quarticity, numpy array of dates (text))
    """

    data['lr4'] = (np.log(data.price) - np.log(data.price.shift(1)))**4
    data = data[data['lr4'].notna()]

    alldays = data['date'].unique()
    nbdays = len(alldays)
    realizeddailylogrange = np.zeros((nbdays,))

    idx=0
    for day, g in data.groupby('date'):
        realizeddailylogrange[idx] = sum(g['lr4'])*len(g['lr4'])/3
        
        # if np.sqrt(realizeddailylogrange[idx]*252)<0.1:
        #     print(g['date'].iloc[0])
        idx+=1

    return realizeddailylogrange, alldays


def lr(data:pd.DataFrame):
    """ 
    This function requires a dataframe with two columns ['high'] and ['low'].
    It is expected that the time series be daily, NOT intraday.
    returns a tuple( numpy array of daily log-range, numpy array of dates (text))
    """

    data['lr2'] = 1/(4*np.log(2))*(np.log(data.high) - np.log(data.low))**2
    data = data[data['lr2'].notna()]

    alldays = data['date'].unique()
    nbdays = len(alldays)
    realizeddailylogrange = np.array(data['lr2'])

    # idx=0
    # for day, g in data.groupby('date'):
    #     realizeddailylogrange[idx] = sum(g['lr2'])
    #     # if np.sqrt(realizeddailylogrange[idx]*252)<0.1:
    #     #     print(g['date'].iloc[0])
    #     idx+=1

    return realizeddailylogrange, alldays
















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