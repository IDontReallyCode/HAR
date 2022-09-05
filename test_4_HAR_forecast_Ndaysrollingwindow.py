import pandas as pd
from HAR import rv
from HAR import HAR
import matplotlib.pyplot as plt
import numpy as np

def main():
    ticker = 'SPY'
    # data = pd.read_csv(f"./intradaysample{ticker}.csv", index_col=0)
    data = pd.read_csv(f"./{ticker}_5m.csv", index_col=0)
    
    # the realized variance requires 2 columns with specific names ['date'] and ['price']
    # ['date'] needs to be just a date. No time.
    data.rename(columns={'nicedate':'date', 'close':'price'}, inplace=True)
    # aggregatesampling = [1,5,10,20]
    aggregatesampling = [1,2,3,4,5, 10, 20]
    horizon = 1
    targettype = HAR.TOTALREALIZEDVARIANCE
    # targettype = HAR.PEAKDREALIZEDVARIANCE

    # split the sample in train and test samples
    # since the time series dynamics is important, don't split randomly...
    # 75% is picked arbitrarily
    # Note that we will NOT train ONLY the initial train data and then forecast everything out-sample
    # We train, we forecast over "horizon" day.
    # We update the betas, we forecast over "horizon" day.
    # We loop until the end of the test sample.

    # [TODO] aggregatesampling needs to have [1] in it, otherwize, the benchmark will be fucked.
    rvdata = HAR.getrvdata(data, aggregatesampling=aggregatesampling)
    nbdays = np.size(rvdata,0)
    rollingwindowsize = 2000
    # Example
    # data = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20]
    # use RW of size 5, forecast 4 days later
    # benchmark = data[rollingwindowsize+horizon-1:,0]
    # benchmark = data[9 10 11 12 13 14 15 16 17 18 19 20] 
    # we are left with len(data)-RW-horizon+1 = 12
    # [TODO] Benchmark NEEDS to match the longer horizon method if greater than 1
    benchmark = rvdata[rollingwindowsize+horizon-1:,0]

    if (horizon>1) and (targettype==HAR.TOTALREALIZEDVARIANCE):
        y = HAR._running_sumba(rvdata[rollingwindowsize:,0], horizon)
    elif (horizon>1) and (targettype==HAR.PEAKDREALIZEDVARIANCE):
        y = HAR._running_maxba(rvdata[rollingwindowsize:,0], horizon)
    else:
        y = rvdata[rollingwindowsize:,0]

    # We need to compare to a benchmark.
    # Here we pick a benchmark where we model the Realized Variance to be the a simple martingale (h_{t+1} = h_{t})
    # So the daily realized variance tomorrow is forecasted to be the realized variance today
    beta_BM = np.zeros((len(aggregatesampling)+1,))
    beta_BM[1] = 1

    HAR__OLS_forecast = np.zeros((nbdays-rollingwindowsize-horizon+1,))
    HAR_WOLS_forecast = np.zeros((nbdays-rollingwindowsize-horizon+1,))
    BM_______forecast = np.zeros((nbdays-rollingwindowsize-horizon+1,))

    for index in range(nbdays-rollingwindowsize-horizon+1):
        # Here we estimate the simple linear model for the HAR
        beta__OLS = HAR.estimateHARols(rvdata[0+index:(rollingwindowsize+index),:], aggregatesampling, forecasthorizon=horizon, longerhorizontype=targettype)
        beta_WOLS = HAR.estimateHARwols(rvdata[0+index:(rollingwindowsize+index),:], aggregatesampling, forecasthorizon=horizon, longerhorizontype=targettype)

        HAR__OLS_forecast[index] = HAR.forecast(rvdata[(rollingwindowsize+index-1),:], beta__OLS)
        HAR_WOLS_forecast[index] = HAR.forecast(rvdata[(rollingwindowsize+index-1),:], beta_WOLS)
        BM_______forecast[index] = HAR.forecast(rvdata[(rollingwindowsize+index-1),:], beta_BM)

        beta_better, better_forecast = HAR.estimatemodel(rvdata[0+index:(rollingwindowsize+index),:], aggregatesampling, forecasthorizon=horizon, longerhorizontype=targettype, 
                                                    datatransformation=HAR.TRANSFORM_DO_NOTHN, estimationmethod=HAR.METHOD_OLS)

        pausehere=1
    
    # benchmark = np.reshape(benchmark,(testsize,1))
    # HAR_Rsquared = np.linalg.lstsq(benchmark,HAR__OLS_forecast,rcond=None)[1]
    # BM__Rsquared = np.linalg.lstsq(benchmark,BM_______forecast,rcond=None)[1]
    corr_matrix = np.corrcoef(y, HAR__OLS_forecast)
    corr = corr_matrix[0,1]
    HAR_R_sq__OLS = corr**2
    corr_matrix = np.corrcoef(y, HAR_WOLS_forecast)
    corr = corr_matrix[0,1]
    HAR_R_sq_WOLS = corr**2
    corr_matrix = np.corrcoef(y, BM_______forecast)
    corr = corr_matrix[0,1]
    BM__R_sq = corr**2

    fig, ax = plt.subplots(1,3, sharey=True)
    ax[0].plot(y,BM_______forecast,'r.', label=f"Mart. R^2={BM__R_sq:0.4f}")
    ax[0].legend()
    ax[0].title.set_text('Basic Martingale Forecast')
    ax[1].plot(y,HAR__OLS_forecast,'b.', label=f"HAR R^2={HAR_R_sq__OLS:0.4f}")
    ax[1].legend()
    ax[1].title.set_text('HAR using OLS Forecast')
    ax[2].plot(y,HAR_WOLS_forecast,'b.', label=f"HAR R^2={HAR_R_sq_WOLS:0.4f}")
    ax[2].legend()
    ax[2].title.set_text('HAR using WOLS Forecast')
    fig.tight_layout()
    fig.suptitle(f"Forecasting over {horizon} days")
    plt.show()

    done=1


#### __name__ MAIN()
if __name__ == '__main__':
    main()


