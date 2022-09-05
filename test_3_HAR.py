import pandas as pd
from HAR import rv
from HAR import HAR
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import numpy as np

def main():
    ticker = 'SPY'
    # data = pd.read_csv(f"./intradaysample{ticker}.csv", index_col=0)
    data = pd.read_csv(f"./{ticker}_5m.csv", index_col=0)
    
    # the realized variance requires 2 columns with specific names ['date'] and ['price']
    # ['date'] needs to be just a date. No time.
    # data.rename(columns={'nicedate':'date', 'close':'price'}, inplace=True)

    # rvdays, realizeddailyvariance = rv.rv(data)

    # multiplesampling = rv.rvaggregate(realizeddailyvariance)
    # X = np.ones((np.size(multiplesampling,0)-1,np.size(multiplesampling,1)+1))
    # X[:,1:] = multiplesampling[0:-1,:]
    # y = multiplesampling[1:,0]

    # # we need to create the X matrix that has a column of 1, then each time horizon RV
    # # the y will be a the daily RV shifted by N days, depending on the time horizon we want to forecast

    # betashere = np.linalg.lstsq(X,y,rcond=None)[0]

    # beta1__OLS = HAR.estimateHARols(data, datecolumnname='nicedate', closingpricecolumnname='close', forecasthorizon=1, longerhorizontype=HAR.PEAKDREALIZEDVARIANCE)
    # beta1_WOLS = HAR.estimateHARwols(data, datecolumnname='nicedate', closingpricecolumnname='close', forecasthorizon=1, longerhorizontype=HAR.PEAKDREALIZEDVARIANCE)
    beta2__OLS, forecast__OLS = HAR.estimatemodel(data, [1,5,20], 'nicedate', 'close', HAR.MODEL_HAR, HAR.TRANSFORM_DO_NOTHN, HAR.METHOD_OLS, 1, HAR.PEAKDREALIZEDVARIANCE)
    beta2_WOLS, forecast_WOLS = HAR.estimatemodel(data, [1,5,20], 'nicedate', 'close', HAR.MODEL_HAR, HAR.TRANSFORM_DO_NOTHN, HAR.METHOD_WOLS, 1, HAR.PEAKDREALIZEDVARIANCE)
    betaQ2_OLS, forecast_QOLS = HAR.estimatemodel(data, [1,5,20], 'nicedate', 'close', HAR.MODEL_HARQ, HAR.TRANSFORM_DO_NOTHN, HAR.METHOD_OLS, 1, HAR.PEAKDREALIZEDVARIANCE)
    betaQ2WOLS, forecastQWOLS = HAR.estimatemodel(data, [1,5,20], 'nicedate', 'close', HAR.MODEL_HARQ, HAR.TRANSFORM_DO_NOTHN, HAR.METHOD_WOLS, 1, HAR.PEAKDREALIZEDVARIANCE)
    betaQLWOLS, forecasQLWOLS = HAR.estimatemodel(data, [1,5,20], 'nicedate', 'close', HAR.MODEL_HARQ, HAR.TRANSFORM_TAKE_LOG, HAR.METHOD_WOLS, 1, HAR.PEAKDREALIZEDVARIANCE)



    done=1


#### __name__ MAIN()
if __name__ == '__main__':
    main()


