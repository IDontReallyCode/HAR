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
    aggregatesampling = [1,5,10,20]

    # split the sample in train and test samples
    # since the time series dynamics is important, don't split randomly...
    # 75% is picked arbitrarily
    # Note that we will NOT train ONLY the initial train data and then forecast everything out-sample
    # We train, we forecast over 1 day.
    # We update the betas, we forecast over 1 more day.
    # We loop until the end of the test sample.

    rvdata = HAR.getrvdata(data, aggregatesampling=aggregatesampling)
    nbdays = np.size(rvdata,0)
    rollingwindowsize = 2000
    benchmark = rvdata[rollingwindowsize:,0]

    # We need to compare to a benchmark.
    # Here we pick a benchmark where we model the Realized Variance to be the a simple martingale (h_{t+1} = h_{t})
    # So the daily realized variance tomorrow is forecasted to be the realized variance today
    beta_BM = np.zeros((len(aggregatesampling)+1,))
    beta_BM[1] = 1

    HAR__OLS_forecast = np.zeros((nbdays-rollingwindowsize,))
    HAR_WOLS_forecast = np.zeros((nbdays-rollingwindowsize,))
    BM_______forecast = np.zeros((nbdays-rollingwindowsize,))

    for index in range(nbdays-rollingwindowsize):
        # Here we estimate the simple linear model for the HAR
        beta__OLS = HAR.estimateHARols(rvdata[0+index:(rollingwindowsize+index-1),:], aggregatesampling)
        beta_WOLS = HAR.estimateHARwols(rvdata[0+index:(rollingwindowsize+index-1),:], aggregatesampling)

        HAR__OLS_forecast[index] = HAR.forecast(rvdata[(rollingwindowsize+index-1),:], beta__OLS)
        HAR_WOLS_forecast[index] = HAR.forecast(rvdata[(rollingwindowsize+index-1),:], beta_WOLS)
        BM_______forecast[index] = HAR.forecast(rvdata[(rollingwindowsize+index-1),:], beta_BM)

    
    # benchmark = np.reshape(benchmark,(testsize,1))
    # HAR_Rsquared = np.linalg.lstsq(benchmark,HAR__OLS_forecast,rcond=None)[1]
    # BM__Rsquared = np.linalg.lstsq(benchmark,BM_______forecast,rcond=None)[1]
    corr_matrix = np.corrcoef(benchmark, HAR__OLS_forecast)
    corr = corr_matrix[0,1]
    HAR_R_sq__OLS = corr**2
    corr_matrix = np.corrcoef(benchmark, HAR_WOLS_forecast)
    corr = corr_matrix[0,1]
    HAR_R_sq_WOLS = corr**2
    corr_matrix = np.corrcoef(benchmark, BM_______forecast)
    corr = corr_matrix[0,1]
    BM__R_sq = corr**2

    fig, ax = plt.subplots(1,3)
    ax[0].plot(benchmark,BM_______forecast,'r.', label=f"Mart. R^2={BM__R_sq:0.4f}")
    ax[0].legend()
    ax[0].title.set_text('Basic Martingale Forecast')
    ax[1].plot(benchmark,HAR__OLS_forecast,'b.', label=f"HAR R^2={HAR_R_sq__OLS:0.4f}")
    ax[1].legend()
    ax[1].title.set_text('HAR using OLS Forecast')
    ax[2].plot(benchmark,HAR_WOLS_forecast,'b.', label=f"HAR R^2={HAR_R_sq_WOLS:0.4f}")
    ax[2].legend()
    ax[2].title.set_text('HAR using WOLS Forecast')
    plt.show()

    done=1


#### __name__ MAIN()
if __name__ == '__main__':
    main()


