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
# from sklearn.linear_model import LinearRegression

MODEL_HAR = 0
MODEL_HARQ = 1
METHOD_OLS = 0
METHOD_WOLS = 1

TOTALREALIZEDVARIANCE = 0
PEAKDREALIZEDVARIANCE = 1

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


def rvaggregate(dailyrv: np.ndarray, aggregatesampling: list=[1,5,20]):
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



@njit
def _running_sumba(x, N):
    # based on https://stackoverflow.com/a/27681394 
    # but avoids using insert only returns the fully summed numbers (output vector is shorter)
    # also numba possible now
    cumsum = np.zeros((len(x)+1,1))
    cumsum[1:,0] = np.cumsum(x)
    cumsum[N:,0] = (cumsum[N:,0] - cumsum[:-N,0])
    return cumsum[N:,0] 



# @njit
def _running_maxba(x, N):
    peaks = np.zeros(len(x)-N+1,)
    for index in range(len(x)-N):
        peaks[index] = max(x[index:index+N])
    return peaks 


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
    realizedquarticity = np.zeros((nbdays,))

    idx=0
    for day, g in data.groupby('date'):
        realizedquarticity[idx] = sum(g['lr4'])*len(g['lr4'])/3
        idx+=1

    return realizedquarticity, alldays


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


def getrvdata(data:pd.DataFrame, aggregatesampling: list=[1,5,10,20], datecolumnname='date', closingpricecolumnname='price'):
    """
        This function uses the pandas Dataframe to calculate the Realized Variance and aggregate of multiple time horizon
    """
    rvdaily = rv(data, datecolumnname, closingpricecolumnname)[0]
    return rvaggregate(rvdaily, aggregatesampling=aggregatesampling)


def estimateHARols(data:Union[np.ndarray, pd.DataFrame], aggregatesampling:list[int]=[1,5,20], 
                    datecolumnname:str='date', closingpricecolumnname:str='price', forecasthorizon:int=1, longerhorizontype:int=TOTALREALIZEDVARIANCE)->np.ndarray:
    """
        This function will estimate_ols the HAR beta coefficients on either the raw pandas.Dataframe, or the aggregated Realized Variance
    """
    # [TODO] : Allow for different horizon than 1day.
    if type(data)==pd.DataFrame:
        realizeddailyvariance = rv(data, datecolumnname, closingpricecolumnname)[0]
        multiplesampling = rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
    else:
        multiplesampling = data
    X = np.ones((np.size(multiplesampling,0)-forecasthorizon,np.size(multiplesampling,1)+1))
    X[:,1:] = multiplesampling[0:-forecasthorizon,:]
    if (forecasthorizon>1) and (longerhorizontype==TOTALREALIZEDVARIANCE):
        y = _running_sumba(multiplesampling[1:,0], forecasthorizon)
    elif (forecasthorizon>1) and (longerhorizontype==PEAKDREALIZEDVARIANCE):
        y = _running_maxba(multiplesampling[1:,0], forecasthorizon)
    else:
        y = multiplesampling[1:,0]

    beta = np.linalg.lstsq(X,y,rcond=None)[0]
    return beta


def estimateHARwols(data:Union[np.ndarray, pd.DataFrame], aggregatesampling: list=[1,5,20], 
                    datecolumnname='date', closingpricecolumnname='price', forecasthorizon:int=1, longerhorizontype:int=TOTALREALIZEDVARIANCE)->np.ndarray:
    """
        This function will estimate_wols the HAR beta coefficients on either the raw pandas.Dataframe, or the aggregated Realized Variance
        using a simple scheme for weights of 1/RV, see Clement and Preve (2021) https://www.sciencedirect.com/science/article/pii/S0378426621002417
    """
    # weighted OLS : https://stackoverflow.com/a/52452833

    if type(data)==pd.DataFrame:
        realizeddailyvariance = rv(data, datecolumnname, closingpricecolumnname)[0]
        multiplesampling = rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
    else:
        multiplesampling = data
    X = np.ones((np.size(multiplesampling,0)-forecasthorizon,np.size(multiplesampling,1)+1))
    X[:,1:] = multiplesampling[0:-forecasthorizon,:]
    if (forecasthorizon>1) and (longerhorizontype==TOTALREALIZEDVARIANCE):
        y = _running_sumba(multiplesampling[1:,0], forecasthorizon)
    elif (forecasthorizon>1) and (longerhorizontype==PEAKDREALIZEDVARIANCE):
        y = _running_maxba(multiplesampling[1:,0], forecasthorizon)
    else:
        y = multiplesampling[1:,0]

    W = 1/multiplesampling[0:-forecasthorizon,0]

    beta = np.linalg.lstsq(X*W[:,None],y*W,rcond=None)[0]
    return beta


def forecast(aggregatedrv, beta):
    # TODO we need to track what aggregatesampling was used.
    X = np.ones((1,len(aggregatedrv)+1))
    X[:,1:] = aggregatedrv

    forecast = np.matmul(X,beta)
    return forecast


def estimateforecast(data:pd.DataFrame, aggregatesampling: list=[1,5,20], datecolumnname='date', closingpricecolumnname='price', method=METHOD_WOLS, horizon:int=1)->dict:
    """
        Submit a pandas Dataframe with one column with "date" as just the date, and "price" for the closing price of the candle
    """
    realizeddailyvariance = rv(data, datecolumnname, closingpricecolumnname)[0]
    multiplesampling = rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
    if method==METHOD_OLS:
        beta = estimateHARols(multiplesampling, aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname, forecasthorizon=horizon)
    elif method==METHOD_WOLS:
        beta = estimateHARwols(multiplesampling, aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname, forecasthorizon=horizon)
    else:
        bigDIC = {'status':'Failed, invalid estimation method.'}
        return bigDIC

    HAR_forecast = forecast(multiplesampling[-horizon,:], beta)

    bigDIC = {'rvaggregate':multiplesampling, 'beta':beta, 'last_annualrvol':np.sqrt(252*realizeddailyvariance[-1]), 'forecast_annualvol':np.sqrt(252*HAR_forecast)}

    return bigDIC


def rollingwindowbacktesting(data:pd.DataFrame, aggregatesampling: list[int]=[1,5,20], 
                            datecolumnname:str='date', closingpricecolumnname:str='price', method=METHOD_WOLS,
                            forecasthorizon:list[int]=[1,5,10], rollingwindowsize:int=2000, model=MODEL_HAR,
                            longerhorizontype:int=TOTALREALIZEDVARIANCE)->dict:
    """
        This function will deal entirely with back testing HAR forecasts and returns metrics
        It will also compare to a benchmark of E[RV_{t}] = RV_{t-1}

        Default HAR model is
        E[RV_{t+n}] = b0 + b1*RV_{t}^{1d} + b2*RV_{t}^{5d} + b3*RV_{t}^{20d}
    """

    # Get the Realized Variance data
    rvdata = getrvdata(data, aggregatesampling)
    totalnbdays = np.size(rvdata,0)
    nbhorizon = len(forecasthorizon)


    output = {}
    for ihor in forecasthorizon:
        # benchmark = rvdata[rollingwindowsize+ihor-1:,0]

        if (ihor>1) and (longerhorizontype==TOTALREALIZEDVARIANCE):
            x = _running_sumba(rvdata[rollingwindowsize:,0], ihor)
        elif (ihor>1) and (longerhorizontype==PEAKDREALIZEDVARIANCE):
            x = _running_maxba(rvdata[rollingwindowsize:,0], ihor)
        else:
            x = rvdata[rollingwindowsize:,0]

        # The benchmark forecast E[RV_{t+n}] = RV_{t}^{1d}
        # i.e. The model is E[RV_{t+n}] = 0 + 1*RV_{t}^{1d} + 0*RV_{t}^{5d} + 0*RV_{t}^{20d}
        beta_bench = np.zeros((len(aggregatesampling)+1,))
        beta_bench[1] = 1
        
        model_forecast = np.zeros((totalnbdays-rollingwindowsize-ihor+1,))
        model_forecast = np.zeros((totalnbdays-rollingwindowsize-ihor+1,))
        bench_forecast = np.zeros((totalnbdays-rollingwindowsize-ihor+1,))
        for index in range(totalnbdays-rollingwindowsize-ihor+1):
            # Here we estimate the simple linear model for the HAR
            if method==METHOD_OLS:
                beta_model = estimateHARols(rvdata[0+index:(rollingwindowsize+index-1),:], aggregatesampling, forecasthorizon=ihor, longerhorizontype=longerhorizontype)
            elif method==METHOD_WOLS:
                beta_model = estimateHARwols(rvdata[0+index:(rollingwindowsize+index-1),:], aggregatesampling, forecasthorizon=ihor, longerhorizontype=longerhorizontype)
            else:
                raise Exception('Please use the CONSTANT METHOD_OLS or METHOD_WOLS.')

            model_forecast[index] = forecast(rvdata[(rollingwindowsize+index-1),:], beta_model)
            bench_forecast[index] = forecast(rvdata[(rollingwindowsize+index-1),:], beta_bench)

        corr_matrix = np.corrcoef(x, model_forecast)
        corr = corr_matrix[0,1]
        model_Rsquare = corr**2
        corr_matrix = np.corrcoef(x, bench_forecast)
        corr = corr_matrix[0,1]
        bench_Rsquare = corr**2

        beta = np.linalg.lstsq(np.reshape(x,(-1,1)),model_forecast,rcond=None)[0]
        yhat = np.matmul(np.reshape(x,(-1,1)),beta)
        SS_Residual = sum((model_forecast-yhat)**2)       
        SS_Total = sum((model_forecast-np.mean(model_forecast))**2)     
        r_squared = 1 - (float(SS_Residual))/SS_Total
        model_ad_r_squared = 1 - (1-r_squared)*(len(x)-1)/(len(x)-len(aggregatesampling)-1)

        beta = np.linalg.lstsq(np.reshape(x,(-1,1)),bench_forecast,rcond=None)[0]
        yhat = np.matmul(np.reshape(x,(-1,1)),beta)
        SS_Residual = sum((bench_forecast-yhat)**2)       
        SS_Total = sum((bench_forecast-np.mean(bench_forecast))**2)     
        bench_ad_r_squared = 1 - (float(SS_Residual))/SS_Total
        # bench_ad_r_squared = 1 - (1-r_squared)*(len(x)-1)/(len(x)-len(aggregatesampling)-1)


        model_RMSE = np.sqrt(np.mean((x-model_forecast)**2))
        bench_RMSE = np.sqrt(np.mean((x-bench_forecast)**2))

        model_AME = np.mean(np.abs(x-model_forecast))
        bench_AME = np.mean(np.abs(x-bench_forecast))

        output[ihor] = {'model':{'Rsquare':model_Rsquare, 'RMSE':model_RMSE, 'AME':model_AME, 'forecast':model_forecast, 'AdjRsquare':model_ad_r_squared},
                    'bench':{'Rsquare':bench_Rsquare, 'RMSE':bench_RMSE, 'AME':bench_AME, 'forecast':bench_forecast, 'AdjRsquare':bench_ad_r_squared},
                    'realized':{'target':x}}

    return output