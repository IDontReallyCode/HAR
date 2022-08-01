import pandas as pd
from HAR import rv
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import numpy as np

def main():
    ticker = 'VOO'
    data = pd.read_csv(f"./intradaysample{ticker}.csv", index_col=0)
    
    # the realized variance requires 2 columns with specific names ['date'] and ['price']
    # ['date'] needs to be just a date. No time.
    data.rename(columns={'nicedate':'date', 'close':'price'}, inplace=True)

    realizeddailyvariance, rvdays = rv.rq(data)

    multiplesampling = rv.rvaggregate(realizeddailyvariance)

    dates = pltd.date2num(rvdays)
    # plt.plot_date(dates, np.sqrt(realizeddailyvariance*252), 'b-')
    # plt.show()

    plt.plot_date(dates, np.sqrt(multiplesampling[:,0]*252), 'b.', label='rq - daily')
    plt.plot_date(dates, np.sqrt(multiplesampling[:,1]*252), 'g.', label='rq - weekly')
    plt.plot_date(dates, np.sqrt(multiplesampling[:,2]*252), 'r.', label='rq - bi-weekly')
    plt.plot_date(dates, np.sqrt(multiplesampling[:,3]*252), 'k.', label='rq - monthly')
    plt.title("Realized Quarticity")
    plt.legend()
    plt.show()

    # data = pd.read_csv(f"./dailysample{ticker}.csv", index_col=0)
    # data.rename(columns={'date_eod':'date'}, inplace=True)

    # realizeddailylogrange, lrdates = rv.lr(data)

    # multiplesampling = rv.rvaggregate(realizeddailylogrange)

    # dates = pltd.date2num(lrdates)
    # # plt.plot_date(dates, np.sqrt(realizeddailyvariance*252), 'b-')
    # # plt.show()

    # plt.plot_date(dates, np.sqrt(multiplesampling[:,0]*252), 'b-', label='lr - daily')
    # plt.plot_date(dates, np.sqrt(multiplesampling[:,1]*252), 'g-', label='lr - weekly')
    # plt.plot_date(dates, np.sqrt(multiplesampling[:,2]*252), 'r-', label='lr - bi-weekly')
    # plt.plot_date(dates, np.sqrt(multiplesampling[:,3]*252), 'k-', label='lr - monthly')
    # plt.title(f"{ticker}, compare RV and log-range")
    # plt.legend()
    # plt.show()





#### __name__ MAIN()
if __name__ == '__main__':
    main()


