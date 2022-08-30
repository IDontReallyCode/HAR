"""
This library will receive a pandas Dataframe with intraday data for a time series
Will "clean" up the data and regularize the time series aggregation
Will calculate the daily realized volatility based on a specified time aggregation (5min default)
Will return the 1-day, 1-week, 2-week, 1-month (and other) time agregation realized volatilities 
Will estimate_ols the HAR forecasting model
"""

from typing import Union
import numpy as np
import pandas as pd
from numba import njit


METHOD_OLS = 0
METHOD_WOLS = 1

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


def rvdata(data:pd.DataFrame, aggregatesampling: list=[1,5,10,20], datecolumnname='date', closingpricecolumnname='price'):
    """
        This function uses the pandas Dataframe to calculate the Realized Variance and aggregate of multiple time horizon
    """
    rvdaily = rv.rv(data, datecolumnname, closingpricecolumnname)[0]
    return rv.rvaggregate(rvdaily, aggregatesampling=aggregatesampling)


def estimate_ols(data:Union[np.ndarray, pd.DataFrame], aggregatesampling: list=[1,5,10,20], datecolumnname='date', closingpricecolumnname='price')->np.ndarray:
    """
        This function will estimate_ols the HAR beta coefficients on either the raw pandas.Dataframe, or the aggregated Realized Variance
    """
    if type(data)==pd.DataFrame:
        realizeddailyvariance = rv.rv(data, datecolumnname, closingpricecolumnname)[0]
        multiplesampling = rv.rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
    else:
        multiplesampling = data
    X = np.ones((np.size(multiplesampling,0)-1,np.size(multiplesampling,1)+1))
    X[:,1:] = multiplesampling[0:-1,:]
    y = multiplesampling[1:,0]

    beta = np.linalg.lstsq(X,y,rcond=None)[0]
    return beta


def estimate_wols(data:Union[np.ndarray, pd.DataFrame], aggregatesampling: list=[1,5,10,20], datecolumnname='date', closingpricecolumnname='price')->np.ndarray:
    """
        This function will estimate_wols the HAR beta coefficients on either the raw pandas.Dataframe, or the aggregated Realized Variance
        using a simple scheme for weights of 1/RV, see Clement and Preve (2021) https://www.sciencedirect.com/science/article/pii/S0378426621002417
    """
    # weighted OLS : https://stackoverflow.com/a/52452833

    if type(data)==pd.DataFrame:
        realizeddailyvariance = rv.rv(data, datecolumnname, closingpricecolumnname)[0]
        multiplesampling = rv.rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
    else:
        multiplesampling = data
    X = np.ones((np.size(multiplesampling,0)-1,np.size(multiplesampling,1)+1))
    X[:,1:] = multiplesampling[0:-1,:]
    y = multiplesampling[1:,0]
    W = 1/multiplesampling[0:-1,0]

    beta = np.linalg.lstsq(X*W[:,None],y*W,rcond=None)[0]
    return beta



def forecast(aggregatedrv, beta):
    # TODO we need to track what aggregatesampling was used.
    X = np.ones((1,len(aggregatedrv)+1))
    X[:,1:] = aggregatedrv

    forecast = np.matmul(X,beta)
    return forecast


def estimateforecast(data:pd.DataFrame, aggregatesampling: list=[1,5,10,20], datecolumnname='date', closingpricecolumnname='price', method=METHOD_WOLS)->dict:
    """
        Submit a pandas Dataframe with one column with "date" as just the date, and "price" for the closing price of the candle
    """
    realizeddailyvariance = rv.rv(data, datecolumnname, closingpricecolumnname)[0]
    multiplesampling = rv.rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
    if method==METHOD_OLS:
        beta = estimate_ols(multiplesampling, aggregatesampling)
    elif method==METHOD_WOLS:
        beta = estimate_wols(multiplesampling, aggregatesampling)
    else:
        bigDIC = {'status':'Failed, invalid estimation method.'}
        return bigDIC

    HAR_forecast = forecast(multiplesampling[-1,:], beta)

    bigDIC = {'rvaggregate':multiplesampling, 'beta':beta, 'last_annualrvol':np.sqrt(252*realizeddailyvariance[-1]), 'forecast_annualvol':np.sqrt(252*HAR_forecast)}

    return bigDIC





