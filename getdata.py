"""
Do not use this program.
Get your own intraday data using your available means.
This program is offered as an example only
"""
from tdalocal import TDA_STUFF as td
import pandas as pd

def main():
    # get n minute data for MSFT
    nminute = 5
    ticker = 'TSLA'
    data = td.Get_Ticker_nMinuteSampling(ticker, nminutesampling=nminute)
    intradaydf = pd.json_normalize(data, record_path=['candles'])
    intradaydf['nicedatetime'] =  intradaydf['datetime'].map(lambda x: td.unix_convert(x,False))
    intradaydf['nicedate'] =  intradaydf['datetime'].map(lambda x: td.unix_convert(x,True))
    
    print(intradaydf)

    intradaydf.to_csv("./intradaysample.csv")

    done = 1
    






#### __name__ MAIN()
if __name__ == '__main__':
    main()


