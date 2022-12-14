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
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

MODEL_HAR = 0
MODEL_HARQ = 1
MODEL_HARM = 2
MODEL_HARMC = 3
TRANSFORM_DO_NOTHN = 0
TRANSFORM_TAKE_LOG = 1
METHOD_OLS = 0
METHOD_WOLS = 1
METHOD_RFR = 2

TOTALREALIZEDVARIANCE = 0
PEAKDREALIZEDVARIANCE = 1

WINDOW_TYPE_ROLLING = 0
WINDOW_TYPE_GROWING = 1

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

    allgood = True

    idx=0
    for day, g in data.groupby(datecolumnname):
        realizeddailylogrange[idx] = sum(g['lr2'])
        if np.sqrt(realizeddailylogrange[idx]*252)<0.01:
            print(f"There is an issue with data. Could be a lack of trading and zero returns, or you did not account for time zones properly. Problem date: {g[datecolumnname].iloc[0]}")
            if idx==0:
                realizeddailylogrange[idx] = (0.2*0.2)/252
            else:
                realizeddailylogrange[idx] = realizeddailylogrange[idx-1]
            allgood = False
        idx+=1

    return realizeddailylogrange, alldays, allgood


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


@njit
def _running_maxba(x, N):
    peaks = np.zeros(len(x)-N+1,)
    for index in range(len(x)-N+1):
        peaks[index] = max(x[index:index+N])
    return peaks 


def rq(data:pd.DataFrame, datecolumnname='date', closingpricecolumnname='price'):
    """ 
    This function requires a dataframe with two columns ['date'] and ['price'].
    The column ['date'] needs to be just a date. No time.
    returns a tuple( numpy array of daily realized quarticity, numpy array of dates (text))
    """

    data['lr4'] = (np.log(data[closingpricecolumnname]) - np.log(data[closingpricecolumnname].shift(1)))**4
    data = data[data['lr4'].notna()]

    alldays = data[datecolumnname].unique()
    nbdays = len(alldays)
    realizedquarticity = np.zeros((nbdays,))

    idx=0
    for day, g in data.groupby(datecolumnname):
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


def getrvdata(data:pd.DataFrame, aggregatesampling: list=[1,5,20], datecolumnname='date', closingpricecolumnname='price'):
    """
        This function uses the pandas Dataframe to calculate the Realized Variance and aggregate of multiple time horizon
    """
    rvdaily = rv(data, datecolumnname, closingpricecolumnname)[0]
    return rvaggregate(rvdaily, aggregatesampling=aggregatesampling)


# def estimateHARols(data:Union[np.ndarray, pd.DataFrame], aggregatesampling:list[int]=[1,5,20], 
#                     datecolumnname:str='date', closingpricecolumnname:str='price', forecasthorizon:int=1, longerhorizontype:int=TOTALREALIZEDVARIANCE)->np.ndarray:
#     """
#         This function will estimate_ols the HAR beta coefficients on either the raw pandas.Dataframe, or the aggregated Realized Variance
#     """
#     # [TODO] : Allow for different horizon than 1day.
#     if type(data)==pd.DataFrame:
#         realizeddailyvariance = rv(data, datecolumnname, closingpricecolumnname)[0]
#         multiplesampling = rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
#     else:
#         multiplesampling = data
#     X = np.ones((np.size(multiplesampling,0)-forecasthorizon,np.size(multiplesampling,1)+1))
#     X[:,1:] = multiplesampling[0:-forecasthorizon,:]
#     if (forecasthorizon>1) and (longerhorizontype==TOTALREALIZEDVARIANCE):
#         y = _running_sumba(multiplesampling[1:,0], forecasthorizon)
#     elif (forecasthorizon>1) and (longerhorizontype==PEAKDREALIZEDVARIANCE):
#         y = _running_maxba(multiplesampling[1:,0], forecasthorizon)
#     else:
#         y = multiplesampling[1:,0]

#     beta = np.linalg.lstsq(X,y,rcond=None)[0]
#     return beta


# def estimateHARwols(data:Union[np.ndarray, pd.DataFrame], aggregatesampling: list=[1,5,20], 
#                     datecolumnname='date', closingpricecolumnname='price', forecasthorizon:int=1, longerhorizontype:int=TOTALREALIZEDVARIANCE)->np.ndarray:
#     """
#         This function will estimate_wols the HAR beta coefficients on either the raw pandas.Dataframe, or the aggregated Realized Variance
#         using a simple scheme for weights of 1/RV, see Clement and Preve (2021) https://www.sciencedirect.com/science/article/pii/S0378426621002417
#     """
#     # weighted OLS : https://stackoverflow.com/a/52452833

#     if type(data)==pd.DataFrame:
#         realizeddailyvariance = rv(data, datecolumnname, closingpricecolumnname)[0]
#         multiplesampling = rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
#     else:
#         multiplesampling = data
#     X = np.ones((np.size(multiplesampling,0)-forecasthorizon,np.size(multiplesampling,1)+1))
#     X[:,1:] = multiplesampling[0:-forecasthorizon,:]
#     if (forecasthorizon>1) and (longerhorizontype==TOTALREALIZEDVARIANCE):
#         y = _running_sumba(multiplesampling[1:,0], forecasthorizon)
#     elif (forecasthorizon>1) and (longerhorizontype==PEAKDREALIZEDVARIANCE):
#         y = _running_maxba(multiplesampling[1:,0], forecasthorizon)
#     else:
#         y = multiplesampling[1:,0]

#     W = 1/multiplesampling[0:-forecasthorizon,0]

#     beta = np.linalg.lstsq(X*W[:,None],y*W,rcond=None)[0]
#     return beta


def estimatemodel(data:Union[np.ndarray, pd.DataFrame], aggregatesampling:list[int]=[1,5,20], 
                    datecolumnname:str='date', closingpricecolumnname:str='price',
                    model:int=MODEL_HAR, datatransformation:int=TRANSFORM_DO_NOTHN, estimationmethod:int=METHOD_OLS, 
                    forecasthorizon:int=1, longerhorizontype:int=TOTALREALIZEDVARIANCE)->np.ndarray:
    """
        Either pass the Dataframe, or pass the numpy array of data,
        where the columns are, e.g., 
        [rv^{1d}, rv^{5d}, rv^{20d}, rq^{1d}]

        Default HAR model is
        E[RV_{t+1}] = b0 + b1*RV_{t}^{1d} + b2*RV_{t}^{5d} + b3*RV_{t}^{20d}
        HOwever, you can change the factors through the "aggregatesampling"

        HARQ model is
        E[RV_{t+1}] = b0 + b1*RV_{t}^{1d} + b2*RV_{t}^{5d} + b3*RV_{t}^{20d} + b4*SQRT[RQ_{t}^{1d}*RV_{t}^{1d}]

        E[RV_{t+1}] can be replace by E[target] for longer time horizon. 
        For example: total variance over N days, or peak variance over N days
    """
    if type(data)==pd.DataFrame:
        realizeddailyvariance = rv(data, datecolumnname, closingpricecolumnname)[0]
        multiplervsampling = rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
        if model in [MODEL_HARQ]:
            realizedquarticity = rq(data, datecolumnname, closingpricecolumnname)[0]
    else:
        if model in [MODEL_HARQ]:
            multiplervsampling = data[:,:-1]
            realizedquarticity = data[:,-1]
        else:
            multiplervsampling = data

    # generate the X matrix for the model factors
    if model in [MODEL_HARQ]:
        X_in = np.ones((np.size(multiplervsampling,0)-forecasthorizon,np.size(multiplervsampling,1)+2))
        Xout = np.ones((1,np.size(multiplervsampling,1)+2))
        X_in[:,1:-1] = multiplervsampling[0:-forecasthorizon,:]
        Xout[:,1:-1] = multiplervsampling[-1,:]
        X_in[:,-1] = realizedquarticity[0:-forecasthorizon]*X_in[:,1]
        Xout[:,-1] = realizedquarticity[-1]*X_in[-1,1]
    elif model in [MODEL_HARM]:
        X_in = np.ones((np.size(multiplervsampling,0)-forecasthorizon,np.size(multiplervsampling,1)+2))
        Xout = np.ones((1,np.size(multiplervsampling,1)+2))
        X_in[:,1:-1] = multiplervsampling[0:-forecasthorizon,:]
        Xout[:,1:-1] = multiplervsampling[-1,:]
        X_in[:,-1] = np.max(multiplervsampling[0:-forecasthorizon,0])
        Xout[:,-1] = np.max(multiplervsampling[:,0])
    elif model==MODEL_HARMC:
        X_in = np.ones((np.size(multiplervsampling,0)-forecasthorizon,np.size(multiplervsampling,1)+2))
        Xout = np.ones((1,np.size(multiplervsampling,1)+2))
        X_in[:,1:-1] = multiplervsampling[0:-forecasthorizon,:]
        Xout[:,1:-1] = multiplervsampling[-1,:]
        X_in[:,-1] = np.max(multiplervsampling[0:-forecasthorizon,0])*(np.size(multiplervsampling,0)-forecasthorizon - np.argmax(multiplervsampling[0:-forecasthorizon,0]))
        Xout[:,-1] = np.max(multiplervsampling[:,1])*(np.size(multiplervsampling,0) - np.argmax(multiplervsampling[:,0]))
    else:
        X_in = np.ones((np.size(multiplervsampling,0)-forecasthorizon,np.size(multiplervsampling,1)+1))
        Xout = np.ones((1,np.size(multiplervsampling,1)+1))
        X_in[:,1:] = multiplervsampling[0:-forecasthorizon,:]
        Xout[:,1:] = multiplervsampling[-1,:]

    # generate the y vector of the variable to "explain"
    if (forecasthorizon>1) and (longerhorizontype==TOTALREALIZEDVARIANCE):
        y = _running_sumba(multiplervsampling[1:,0], forecasthorizon)
    elif (forecasthorizon>1) and (longerhorizontype==PEAKDREALIZEDVARIANCE):
        y = _running_maxba(multiplervsampling[1:,0], forecasthorizon)
    else:
        y = multiplervsampling[1:,0]

    if datatransformation==TRANSFORM_TAKE_LOG:
        X_in[:,1:] = np.log(X_in[:,1:])
        Xout[:,1:] = np.log(Xout[:,1:])
        y = np.log(y)

    if estimationmethod==METHOD_WOLS:
        W = 1/multiplervsampling[0:-forecasthorizon,0]
        beta = np.linalg.lstsq(X_in*W[:,None],y*W,rcond=None)[0]
        return beta, np.matmul(Xout,beta), np.matmul(X_in,beta)
    elif estimationmethod==METHOD_OLS:
        beta = np.linalg.lstsq(X_in,y,rcond=None)[0]
        return beta, np.matmul(Xout,beta), np.matmul(X_in,beta)
    elif estimationmethod==METHOD_RFR:
        ensemblesize = 1000
        njobs = 24
        minsampleleaf = 5
        ensemble = RandomForestRegressor(n_estimators=ensemblesize, n_jobs=njobs, min_samples_leaf=minsampleleaf)
        ensemble.fit(X_in, y)
        predictions = ensemble.predict(Xout)
        return ensemble.feature_importances_, predictions

    else:
        raise Exception('This is not a valid estimation method for now!!! Please use METHOD_OLS or METHOD_WOLS')

    # pausehere=1
    # return beta, np.matmul(Xout,beta)


def forecast(aggregatedrv, beta):
    # TODO we need to track what aggregatesampling was used.
    X = np.ones((1,len(aggregatedrv)+1))
    X[:,1:] = aggregatedrv

    forecast = np.matmul(X,beta)
    return forecast


# def estimateforecast(data:pd.DataFrame, aggregatesampling: list=[1,5,20], datecolumnname='date', closingpricecolumnname='price', method=METHOD_WOLS, horizon:int=1)->dict:
#     """
#         Submit a pandas Dataframe with one column with "date" as just the date, and "price" for the closing price of the candle
#     """
#     realizeddailyvariance = rv(data, datecolumnname, closingpricecolumnname)[0]
#     multiplesampling = rvaggregate(realizeddailyvariance, aggregatesampling=aggregatesampling)
#     if method==METHOD_OLS:
#         beta = estimateHARols(multiplesampling, aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname, forecasthorizon=horizon)
#     elif method==METHOD_WOLS:
#         beta = estimateHARwols(multiplesampling, aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname, forecasthorizon=horizon)
#     else:
#         bigDIC = {'status':'Failed, invalid estimation method.'}
#         return bigDIC

#     HAR_forecast = forecast(multiplesampling[-horizon,:], beta)

#     bigDIC = {'rvaggregate':multiplesampling, 'beta':beta, 'last_annualrvol':np.sqrt(252*realizeddailyvariance[-1]), 'forecast_annualvol':np.sqrt(252*HAR_forecast)}

#     return bigDIC


def backtesting(data:pd.DataFrame, aggregatesampling: list[int]=[1,5,20], 
                            datecolumnname:str='date', closingpricecolumnname:str='price', 
                            windowtype:int=WINDOW_TYPE_ROLLING, estimatewindowsize:int=2000, 
                            model:int=MODEL_HAR, datatransformation:int=TRANSFORM_DO_NOTHN, estimationmethod:int=METHOD_OLS, 
                            forecasthorizon:Union[int, np.ndarray]=1, longerhorizontype:int=TOTALREALIZEDVARIANCE)->dict:
    """
        This function will deal entirely with back testing HAR forecasts and returns metrics
        It will also compare to a benchmark of E[RV_{t}] = RV_{t-1}

        Default HAR model is
        E[RV_{t+1}] = b0 + b1*RV_{t}^{1d} + b2*RV_{t}^{5d} + b3*RV_{t}^{20d}
        HOwever, you can change the factors through the "aggregatesampling"

        HARQ model is
        E[RV_{t+1}] = b0 + b1*RV_{t}^{1d} + b2*RV_{t}^{5d} + b3*RV_{t}^{20d} + b4*SQRT[RQ_{t}^{1d}*RV_{t}^{1d}]

        E[RV_{t+1}] can be replace by E[target] for longer time horizon. 
        For example: total variance over N days, or peak variance over N days
    """

    # Get the Realized Variance data
    if model in [MODEL_HARQ]:
        temp = getrvdata(data, aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname)
        rvdata = np.zeros((np.size(temp,0), len(aggregatesampling)+1))
        rvdata[:,:-1] = temp
        rvdata[:,-1] = rq(data, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname)[0]
    elif model in [MODEL_HARM]:
        temp = getrvdata(data, aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname)
        rvdata = np.zeros((np.size(temp,0), len(aggregatesampling)+1))
        rvdata[:,:-1] = temp
        rvdata[:,-1] = np.max(temp[:,0])
    elif model==MODEL_HARMC:
        temp = getrvdata(data, aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname)
        rvdata = np.zeros((np.size(temp,0), len(aggregatesampling)+1))
        rvdata[:,:-1] = temp
        rvdata[:,-1] = np.max(temp[:,0])*(np.size(temp,0) - np.argmax(temp[:,0]))
    else:
        rvdata = getrvdata(data, aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname)

    totalnbdays = np.size(rvdata,0)
    nbhorizon = len(forecasthorizon)


    output = {}
    for ihor in forecasthorizon:
        # benchmark = rvdata[rollingwindowsize+ihor-1:,0]

        if (ihor>1) and (longerhorizontype==TOTALREALIZEDVARIANCE):
            x = _running_sumba(rvdata[estimatewindowsize:,0], ihor)
        elif (ihor>1) and (longerhorizontype==PEAKDREALIZEDVARIANCE):
            x = _running_maxba(rvdata[estimatewindowsize:,0], ihor)
        else:
            x = rvdata[estimatewindowsize:,0]

        # The benchmark forecast E[RV_{t+n}] = RV_{t}^{1d}
        # i.e. The model is E[RV_{t+n}] = 0 + 1*RV_{t}^{1d} + 0*RV_{t}^{5d} + 0*RV_{t}^{20d}
        
        model_forecast = np.zeros((totalnbdays-estimatewindowsize-ihor+1,))
        bench_forecast = np.zeros((totalnbdays-estimatewindowsize-ihor+1,))
        for index in range(totalnbdays-estimatewindowsize-ihor+1):
            # Here we estimate the simple linear model for the HAR
            if windowtype==WINDOW_TYPE_ROLLING:
                beta_model, model_forecast[index] = estimatemodel(data=rvdata[0+index:(estimatewindowsize+index),:], aggregatesampling=aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname,
                                            model=model, datatransformation=datatransformation, estimationmethod=estimationmethod,
                                            forecasthorizon=ihor, longerhorizontype=longerhorizontype)
            elif windowtype==WINDOW_TYPE_GROWING:
                beta_model, model_forecast[index] = estimatemodel(data=rvdata[:(estimatewindowsize+index),:], aggregatesampling=aggregatesampling, datecolumnname=datecolumnname, closingpricecolumnname=closingpricecolumnname,
                                            model=model, datatransformation=datatransformation, estimationmethod=estimationmethod,
                                            forecasthorizon=ihor, longerhorizontype=longerhorizontype)
            else:
                raise Exception('Invalid window type, please use a proper CONSTANT')

            bench_forecast[index] = rvdata[(estimatewindowsize+index-1),0]

        # un-TRANSFORM the potentially tesformed data...
        if datatransformation==TRANSFORM_TAKE_LOG:
            model_forecast = np.exp(model_forecast)
            # bench_forecast = np.exp(bench_forecast)

        model_Rsquare = metrics.r2_score(x, model_forecast)
        model_RMSE = np.sqrt(metrics.mean_squared_error(x, model_forecast))
        model_evs = metrics.explained_variance_score(x, model_forecast)
        model_mae = metrics.mean_absolute_error(x, model_forecast)
        model_mape = metrics.mean_absolute_percentage_error(x, model_forecast)

        bench_Rsquare = metrics.r2_score(x, bench_forecast)
        bench_RMSE = np.sqrt(metrics.mean_squared_error(x, bench_forecast))
        bench_evs = metrics.explained_variance_score(x, bench_forecast)
        bench_mae = metrics.mean_absolute_error(x, bench_forecast)
        bench_mape = metrics.mean_absolute_percentage_error(x, bench_forecast)

        output[ihor] = {'model':{'Rsquare':model_Rsquare, 'RMSE':model_RMSE, 'evs':model_evs, 'forecast':model_forecast, 'mae':model_mae, 'mape':model_mape},
                    'bench':{'Rsquare':bench_Rsquare, 'RMSE':bench_RMSE, 'evs':bench_evs, 'forecast':bench_forecast, 'mae':bench_mae, 'mape':bench_mape},
                    'realized':{'target':x}}

    return output















