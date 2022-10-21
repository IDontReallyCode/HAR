import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as pltd

def main():
    ticker = 'SPY'
    data = pd.read_csv(f"./{ticker}.csv", index_col=0)
    data['fulldate'] = pd.to_datetime(data['tdate'])
    # https://stackoverflow.com/a/42826430
    data['fulldate'] = data['fulldate'].dt.tz_localize('UTC').dt.tz_convert("US/Eastern")
    data['nicedate'] = data['fulldate'].dt.date
    data.drop(['total_volume', 'avg_trade_size', 'time_beg', 'vwap', 'opening_price', 'tick_vwap', 'time_end'], axis=1, inplace=True)
    data1min = data.resample('1min', on='fulldate', closed='right').agg({'tick_open': 'first', 
                                                        'tick_high': 'max', 
                                                        'tick_low': 'min', 
                                                        'tick_close': 'last'}).dropna()
    data1min['lr2at1min'] = (np.log(data1min['tick_close']) - np.log(data1min['tick_close'].shift(1)))**2
    data1min = data1min[data1min['lr2at1min'].notna()]

    data5min = data1min.resample('5min').agg({'tick_open': 'first', 
                                                        'tick_high': 'max', 
                                                        'tick_low': 'min', 
                                                        'tick_close': 'last',
                                                        'lr2at1min':'sum'}).dropna()
    data5min['lr2at5min'] = (np.log(data5min['tick_close']) - np.log(data5min['tick_close'].shift(1)))**2
    dates = pltd.date2num(data5min.index)
    plt.plot_date(dates, np.sqrt(data5min['lr2at1min']*252*84), 'r-')
    # plt.show()

    data1day = data5min.resample('1d').agg({'tick_open': 'first', 
                                                        'tick_high': 'max', 
                                                        'tick_low': 'min', 
                                                        'tick_close': 'last',
                                                        'lr2at5min':'sum'}).dropna()
    dates = pltd.date2num(data1day.index)
    plt.plot_date(dates, np.sqrt(data1day['lr2at5min']*252), 'b-')
    plt.show()
    # data5min.rename(columns={'tick_close':'close'}, inplace=True)

    # data1min['dateevery5'] = data1min.index
    # data1min['remainder'] = (data1min['dateevery5'].dt.minute % 5)
    # data1min['dateevery5'] = data1min['dateevery5'].apply(pd.Timedelta(minutes = pd.to_numeric(data1min['remainder'])))
    # data5min.to_csv(f"{ticker}_5m.csv")
    done=1








#### __name__ MAIN()
if __name__ == '__main__':
    main()


