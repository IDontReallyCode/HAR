import pandas as pd
import HAR
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import numpy as np

def main():
    ticker = 'VOO'
    data = pd.read_csv(f"./intradaysample{ticker}.csv", index_col=0)
    
    # the realized variance requires 2 columns with specific names ['date'] and ['price']
    # ['date'] needs to be just a date. No time.
    data.rename(columns={'nicedate':'date', 'close':'price'}, inplace=True)

    realizeddailyquarticity, rqdays = HAR.rq(data)

    # multiplesampling = HAR.rvaggregate(realizeddailyquarticity)

    dates = pltd.date2num(rqdays)
    # plt.plot_date(dates, np.sqrt(realizeddailyvariance*252), 'b-')
    # plt.show()

    fig, axes = plt.subplots(4,1)

    axes[0].plot_date(dates, realizeddailyquarticity)
    axes[0].set_title('Quarticity')
    axes[1].plot_date(dates, np.log(realizeddailyquarticity))
    axes[0].set_title('log-Quarticity')
    axes[2].plot_date(dates, np.sqrt(realizeddailyquarticity))
    axes[0].set_title('sqrt-Quarticity')
    axes[3].plot_date(dates, np.log(np.sqrt(realizeddailyquarticity)))
    axes[0].set_title('log-sqrt-Quarticity')
    fig.tight_layout()
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


