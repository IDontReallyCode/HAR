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
    aggregatesampling = [1,5,20]
    horizons = [1,5,10,20]
    # target = HAR.PEAKDREALIZEDVARIANCE
    # model = HAR.MODEL_HARQ
    # model = HAR.MODEL_HAR
    model = HAR.MODEL_HARM
    # datatransform = HAR.TRANSFORM_TAKE_LOG
    datatransform = HAR.TRANSFORM_DO_NOTHN
    estimationmethod = HAR.METHOD_WOLS
    # estimationmethod = HAR.METHOD_RFR
    ndaystoestimate = 2520
    mywindowtype = HAR.WINDOW_TYPE_GROWING
    target = HAR.TOTALREALIZEDVARIANCE
    # target = HAR.PEAKDREALIZEDVARIANCE

    results = HAR.backtesting(data=data, aggregatesampling=aggregatesampling, 
                            datecolumnname='date', closingpricecolumnname='price', 
                            windowtype=mywindowtype, estimatewindowsize=ndaystoestimate, 
                            model=model, datatransformation=datatransform, estimationmethod=estimationmethod, 
                            forecasthorizon=horizons, longerhorizontype=target)

    # fig, ax = plt.subplots(1,3, sharey=True)
    # ax[0].plot(benchmark,BM_______forecast,'r.', label=f"Mart. R^2={BM__R_sq:0.4f}")
    # ax[0].legend()
    # ax[0].title.set_text('Basic Martingale Forecast')
    # ax[1].plot(benchmark,HAR__OLS_forecast,'b.', label=f"HAR R^2={HAR_R_sq__OLS:0.4f}")
    # ax[1].legend()
    # ax[1].title.set_text('HAR using OLS Forecast')
    # ax[2].plot(benchmark,HAR_WOLS_forecast,'b.', label=f"HAR R^2={HAR_R_sq_WOLS:0.4f}")
    # ax[2].legend()
    # ax[2].title.set_text('HAR using WOLS Forecast')
    # fig.tight_layout()
    # fig.suptitle(f"Forecasting over {horizon} days")
    # plt.show()

    minT = 0
    maxT = -1

    fig, axes = plt.subplots(2,2)
    # axes[0,0].plot( results[1]['realized']['target'], results[1]['model']['forecast'],'b.', label='HAR_WOLS')
    # axes[0,0].plot( results[1]['realized']['target'], results[1]['bench']['forecast'],'r.', label='Martingale')
    axes[0,0].plot( np.sqrt(results[1]['realized']['target'][minT:maxT]*252), label='REAL')
    axes[0,0].plot( np.sqrt(results[1]['model']['forecast'][minT:maxT]*252), label='HAR_WOLS')
    axes[0,0].plot( np.sqrt(results[1]['bench']['forecast'][minT:maxT]*252), label='bench')
    # axes[0,0].set_title(f"1-day forecast: HAR R^2={results[1]['model']['AdjRsquare']:0.4f}, Bench R^2={results[1]['bench']['AdjRsquare']:0.4f}")
    axes[0,0].set_title(f"1-day forecast: HAR RMSE={results[1]['model']['RMSE']:0.2E}, Bench RMSE={results[1]['bench']['RMSE']:0.2E}")
    # axes[0,0].set_title(f"1-day forecast:   HAR R^2={results[1]['model']['Rsquare']:0.4f},   HAR RMSE={results[1]['model']['RMSE']:0.4f}\n"+
    #                     f"                Bench R^2={results[1]['bench']['Rsquare']:0.4f}, Bench RMSE={results[1]['bench']['RMSE']:0.4f} ")
    axes[0,0].legend()
    # axes[0,1].plot( results[5]['realized']['target'], results[5]['model']['forecast'],'b.', label='HAR_WOLS')
    # axes[0,1].plot( results[5]['realized']['target'], results[5]['bench']['forecast'],'r.', label='Martingale')
    axes[0,1].plot( np.sqrt(results[5]['realized']['target'][minT:maxT]*252), label='REAL')
    axes[0,1].plot( np.sqrt(results[5]['model']['forecast'][minT:maxT]*252), label='HAR_WOLS')
    axes[0,1].plot( np.sqrt(results[5]['bench']['forecast'][minT:maxT]*252*5), label='bench')
    # axes[0,1].set_title(f"5-day forecast: HAR R^2={results[5]['model']['AdjRsquare']:0.4f}, Bench R^2={results[5]['bench']['AdjRsquare']:0.4f}")
    axes[0,1].set_title(f"5-day forecast: HAR RMSE={results[5]['model']['RMSE']:0.2E}, Bench RMSE={results[5]['bench']['RMSE']:0.2E}")
    axes[0,1].legend()
    # axes[1,0].plot(results[10]['realized']['target'],results[10]['model']['forecast'],'b.', label='HAR_WOLS')
    # axes[1,0].plot(results[10]['realized']['target'],results[10]['bench']['forecast'],'r.', label='Martingale')
    axes[1,0].plot( np.sqrt(results[10]['realized']['target'][minT:maxT]*252), label='REAL')
    axes[1,0].plot( np.sqrt(results[10]['model']['forecast'][minT:maxT]*252), label='HAR_WOLS')
    axes[1,0].plot( np.sqrt(results[10]['bench']['forecast'][minT:maxT]*252*10), label='bench')
    # axes[1,0].set_title(f"10-day forecast: HAR R^2={results[10]['model']['AdjRsquare']:0.4f}, Bench R^2={results[10]['bench']['AdjRsquare']:0.4f}")
    axes[1,0].set_title(f"10-day forecast: HAR RMSE={results[10]['model']['RMSE']:0.2E}, Bench RMSE={results[10]['bench']['RMSE']:0.2E}")
    axes[1,0].legend()
    # axes[1,1].plot(results[20]['realized']['target'],results[20]['model']['forecast'],'b.', label='HAR_WOLS')
    # axes[1,1].plot(results[20]['realized']['target'],results[20]['bench']['forecast'],'r.', label='Martingale')
    axes[1,1].plot( np.sqrt(results[20]['realized']['target'][minT:maxT]*252), label='REAL')
    axes[1,1].plot( np.sqrt(results[20]['model']['forecast'][minT:maxT]*252), label='HAR_WOLS')
    axes[1,1].plot( np.sqrt(results[20]['bench']['forecast'][minT:maxT]*252*20), label='bench')
    # axes[1,1].set_title(f"20-day forecast: HAR R^2={results[20]['model']['AdjRsquare']:0.4f}, Bench R^2={results[20]['bench']['AdjRsquare']:0.4f}")
    axes[1,1].set_title(f"20-day forecast: HAR RMSE={results[20]['model']['RMSE']:0.2E}, Bench RMSE={results[20]['bench']['RMSE']:0.2E}")
    axes[1,1].legend()
    fig.tight_layout()
    if target==HAR.PEAKDREALIZEDVARIANCE:
        fig.suptitle('Forecasting PEAK realized variance\n Benchmark is a Martingale forecast')
    elif target==HAR.TOTALREALIZEDVARIANCE:
        fig.suptitle('Forecasting TOTAL realized variance\n Benchmark is a Martingale forecast')
    plt.show()

    done=1


#### __name__ MAIN()
if __name__ == '__main__':
    main()


