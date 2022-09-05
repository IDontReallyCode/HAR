import pandas as pd
import HAR
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

    realizeddailyvariance, rvdays = HAR.rv(data, datecolumnname='nicedate', closingpricecolumnname='close')

    # Cheap trick here to get rid of "off" data that gets binned into weekends
    # rvdays = rvdays[realizeddailyvariance>1e-5]
    # realizeddailyvariance = realizeddailyvariance[realizeddailyvariance>1e-5]

    multiplesampling = HAR.rvaggregate(realizeddailyvariance)

    dates = pltd.date2num(rvdays)
    # plt.plot_date(dates, np.sqrt(realizeddailyvariance*252), 'b-')
    # plt.show()

    fig, axes = plt.subplots(4,1)

    axes[0].plot_date(dates, np.sqrt(multiplesampling[:,0]*252), 'b-', label='rv - daily')
    axes[0].plot_date(dates, np.sqrt(multiplesampling[:,1]*252), 'g-', label='rv - weekly')
    axes[0].plot_date(dates, np.sqrt(multiplesampling[:,2]*252), 'k-', label='rv - monthly')
    axes[0].set_title('Annualized volatility')

    axes[1].plot_date(dates, np.log(multiplesampling[:,0]*252), 'b-', label='rv - daily')
    axes[1].plot_date(dates, np.log(multiplesampling[:,1]*252), 'g-', label='rv - weekly')
    axes[1].plot_date(dates, np.log(multiplesampling[:,2]*252), 'k-', label='rv - monthly')
    axes[1].set_title('Annualized log-variance')

    axes[2].plot_date(dates, np.log(np.sqrt((multiplesampling[:,0]*252))), 'b-', label='rv - daily')
    axes[2].plot_date(dates, np.log(np.sqrt((multiplesampling[:,1]*252))), 'g-', label='rv - weekly')
    axes[2].plot_date(dates, np.log(np.sqrt((multiplesampling[:,2]*252))), 'k-', label='rv - monthly')
    axes[2].set_title('Annualized log-volatility')

    axes[3].plot_date(dates, (multiplesampling[:,0]*252), 'b-', label='rv - daily')
    axes[3].plot_date(dates, (multiplesampling[:,1]*252), 'g-', label='rv - weekly')
    axes[3].plot_date(dates, (multiplesampling[:,2]*252), 'k-', label='rv - monthly')
    axes[3].set_title('Annualized variance')

    fig.tight_layout()
    # plt.title(ticker)
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


