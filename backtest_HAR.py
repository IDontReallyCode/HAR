import pandas as pd
from HAR import rv
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import numpy as np

def main():
    data = pd.read_csv("./intradaysample.csv", index_col=0)
    
    # the realized variance requires 2 columns with specific names ['date'] and ['price']
    # ['date'] needs to be just a date. No time.
    data.rename(columns={'nicedate':'date', 'close':'price'}, inplace=True)

    rvdays, realizeddailyvariance = rv.rv(data)

    multiplesampling = rv.rvaggregate(realizeddailyvariance)
    X = np.ones((np.size(multiplesampling,0)-1,np.size(multiplesampling,1)+1))
    X[:,1:] = multiplesampling[0:-1,:]
    y = multiplesampling[1:,0]

    # we need to create the X matrix that has a column of 1, then each time horizon RV
    # the y will be a the daily RV shifted by N days, depending on the time horizon we want to forecast

    olsresults = np.linalg.lstsq(X,y,rcond=None)

    # dates = pltd.date2num(rvdays)
    # # plt.plot_date(dates, np.sqrt(realizeddailyvariance*252), 'b-')
    # # plt.show()

    # plt.plot_date(dates, np.sqrt(multiplesampling[0]*252), 'b-', label='daily')
    # plt.plot_date(dates, np.sqrt(multiplesampling[1]*252), 'g-', label='weekly')
    # plt.plot_date(dates, np.sqrt(multiplesampling[2]*252), 'r-', label='bi-weekly')
    # plt.plot_date(dates, np.sqrt(multiplesampling[3]*252), 'k-', label='monthly')
    # plt.legend()
    # plt.show()


    done=1


#### __name__ MAIN()
if __name__ == '__main__':
    main()


