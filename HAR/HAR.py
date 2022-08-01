"""
This library will receive a pandas Dataframe with intraday data for a time series
Will "clean" up the data and regularize the time series aggregation
Will calculate the daily realized volatility based on a specified time aggregation (5min default)
Will return the 1-day, 1-week, 2-week, 1-month (and other) time agregation realized volatilities 
Will estimate the HAR forecasting model
"""

from typing import Union
import numpy as np
import pandas as pd
from HAR import rv

def rvdata(data:pd.DataFrame, aggregatesampling: list=[1,5,10,20]):
    """
        This function uses the pandas Dataframe to calculate the Realized Variance and aggregate of multiple time horizon
    """
    rvdaily = rv.rv(data)[1]
    return rv.rvaggregate(rvdaily, aggregatesampling=aggregatesampling)

def estimate(data:Union[np.ndarray, pd.DataFrame], aggregatesampling: list=[1,5,10,20])->np.ndarray:
    """
        This function will estimate the HAR beta coefficients on either the raw pandas.Dataframe, or the aggregated Realized Variance
    """
    if type(data)==pd.DataFrame:
        realizeddailyvariance = rv.rv(data)[1]
        multiplesampling = rv.rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
    else:
        multiplesampling = data
    X = np.ones((np.size(multiplesampling,0)-1,np.size(multiplesampling,1)+1))
    X[:,1:] = multiplesampling[0:-1,:]
    y = multiplesampling[1:,0]

    beta = np.linalg.lstsq(X,y,rcond=None)[0]
    return beta


def forecast(aggregatedrv, beta):
    # TODO we need to track what aggregatesampling was used.
    X = np.ones((1,len(aggregatedrv)+1))
    X[:,1:] = aggregatedrv

    forecast = np.matmul(X,beta)
    return forecast
