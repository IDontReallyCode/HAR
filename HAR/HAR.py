"""
This library will receive a pandas Dataframe with intraday data for a time series
Will "clean" up the data and regularize the time series aggregation
Will calculate the daily realized volatility based on a specified time aggregation (5min default)
Will return the 1-day, 1-week, 2-week, 1-month (and other) time agregation realized volatilities 
Will estimate the HAR forecasting model
"""

import numpy as np
import pandas as pd
from HAR import rv

def estimate(data:pd.DataFrame, aggregatesampling: list=[1,5,10,20]):
    rvdays, realizeddailyvariance = rv.rv(data)

    multiplesampling = rv.rvaggregate(realizeddailyvariance)
    X = np.ones((np.size(multiplesampling,0)-1,np.size(multiplesampling,1)+1))
    X[:,1:] = multiplesampling[0:-1,:]
    y = multiplesampling[1:,0]

    beta = np.linalg.lstsq(X,y,rcond=None)[0]
    return beta



