import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates

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

    return alldays, realizeddailyvariance

def rvaggregate(dailyrv: np.ndarray, aggregatesampling: list=[1,5,10,20]):
    """
    convenient function to aggregate the realized variance at various time horizon
    returns one list of numpy vectors. One vector for each time horizon
    """
    aggregated = [None]*len(aggregatesampling)
    for index, sampling in enumerate(aggregatesampling):
        aggregated[index] = _running_mean(dailyrv,sampling)

    return aggregated


def _running_mean(x, N):
    # https://stackoverflow.com/a/27681394 
    # I added my own twist to keep the variables the same length
    cumsum = np.cumsum(np.insert(x, 0, 0))
    padding = cumsum[1:N] / np.arange(N)[1:N]
    movavg = (cumsum[N:] - cumsum[:-N]) / float(N)
    return np.insert(movavg, 0, padding)