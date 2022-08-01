import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates
from numba import njit

def rv(data:pd.DataFrame):
    """ 
    This function requires a dataframe with two columns ['date'] and ['price'].
    The column ['date'] needs to be just a date. No time.
    returns a tuple( numpy array of dates (text), numpy array of daily realized variance)
    """

    data['lr2'] = (np.log(data.price) - np.log(data.price.shift(1)))**2
    data = data[data['lr2'].notna()]

    alldays = data['date'].unique()
    nbdays = len(alldays)
    realizeddailyvariance = np.zeros((nbdays,))

    idx=0
    for day, g in data.groupby('date'):
        realizeddailyvariance[idx] = sum(g['lr2'])
        # if np.sqrt(realizeddailyvariance[idx]*252)<0.1:
        #     print(g['date'].iloc[0])
        idx+=1

    return realizeddailyvariance, alldays


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
    returns a tuple( numpy array of dates (text), numpy array of daily realized quarticity)
    """

    data['lr4'] = (np.log(data.price) - np.log(data.price.shift(1)))**4
    data = data[data['lr4'].notna()]

    alldays = data['date'].unique()
    nbdays = len(alldays)
    realizeddailyvariance = np.zeros((nbdays,))

    idx=0
    for day, g in data.groupby('date'):
        realizeddailyvariance[idx] = sum(g['lr4'])*len(g['lr4'])/3
        
        # if np.sqrt(realizeddailyvariance[idx]*252)<0.1:
        #     print(g['date'].iloc[0])
        idx+=1

    return alldays, realizeddailyvariance

















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