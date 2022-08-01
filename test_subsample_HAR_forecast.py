import pandas as pd
from HAR import rv
from HAR import HAR
import matplotlib.pyplot as plt
import numpy as np

def main():
    ticker = 'VOO'
    data = pd.read_csv(f"./intradaysample{ticker}.csv", index_col=0)
    
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

    rvdata = HAR.rvdata(data, aggregatesampling=aggregatesampling)
    trainsize = 0.75
    cutoffpnt = int(trainsize*np.size(rvdata,0))
    testsize = np.size(rvdata,0)-cutoffpnt
    benchmark = rvdata[cutoffpnt:,0]

    # We need to compare to a benchmark.
    # Here we pick a benchmark where we model the Realized Variance to be the a simple martingale (h_{t+1} = h_{t})
    # So the daily realized variance tomorrow is forecasted to be the realized variance today
    beta_BM = np.zeros((len(aggregatesampling)+1,))
    beta_BM[1] = 1

    HAR_forecast = np.zeros((testsize,))
    BM__forecast = np.zeros((testsize,))

    for index in range(testsize):
        # Here we estimate the simple linear model for the HAR
        betaHAR = HAR.estimate(rvdata[:(cutoffpnt+index-1),:], aggregatesampling)

        HAR_forecast[index] = HAR.forecast(rvdata[(cutoffpnt+index-1),:], betaHAR)
        BM__forecast[index] = HAR.forecast(rvdata[(cutoffpnt+index-1),:], beta_BM)

    
    # benchmark = np.reshape(benchmark,(testsize,1))
    # HAR_Rsquared = np.linalg.lstsq(benchmark,HAR_forecast,rcond=None)[1]
    # BM__Rsquared = np.linalg.lstsq(benchmark,BM__forecast,rcond=None)[1]
    corr_matrix = np.corrcoef(benchmark, HAR_forecast)
    corr = corr_matrix[0,1]
    HAR_R_sq = corr**2
    corr_matrix = np.corrcoef(benchmark, BM__forecast)
    corr = corr_matrix[0,1]
    BM__R_sq = corr**2

    fig, ax = plt.subplots(1,2)
    ax[0].plot(benchmark,HAR_forecast,'b.', label=f"HAR R^2={HAR_R_sq:0.4f}")
    ax[0].legend()
    ax[1].plot(benchmark,BM__forecast,'r.', label=f"Mart. R^2={BM__R_sq:0.4f}")
    ax[1].legend()
    plt.show()

    done=1


#### __name__ MAIN()
if __name__ == '__main__':
    main()


