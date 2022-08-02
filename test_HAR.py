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

    betat_OLS = HAR.estimate_ols(data, datecolumnname='nicedate', closingpricecolumnname='close')
    betatWOLS = HAR.estimate_wols(data, datecolumnname='nicedate', closingpricecolumnname='close')

    done=1


#### __name__ MAIN()
if __name__ == '__main__':
    main()


