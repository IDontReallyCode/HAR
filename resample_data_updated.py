import pandas as pd
import datetime

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
    data2min = data.resample('10min', on='fulldate', closed='right').agg({'tick_open': 'first', 
                                                        'tick_high': 'max', 
                                                        'tick_low': 'min', 
                                                        'tick_close': 'last'}).dropna()
    data5min = data.resample('5min', on='fulldate').last().dropna()
    data5min.rename(columns={'tick_close':'close'}, inplace=True)

    # data5min.to_csv(f"{ticker}_5m.csv")
    done=1








#### __name__ MAIN()
if __name__ == '__main__':
    main()


