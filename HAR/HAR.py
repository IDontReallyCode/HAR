"""
This library will receive a pandas Dataframe with intraday data for a time series
Will "clean" up the data and regularize the time series aggregation
Will calculate the daily realized volatility based on a specified time aggregation (5min default)
Will return the 1-day, 1-week, 2-week, 1-month (and other) time agregation realized volatilities 
Will estimate the HAR forecasting model
"""

import numpy as np
import pandas as pd
import rv

def estimate(data:pd.DataFrame, aggregatesampling: list=[1,5,10,20]):



