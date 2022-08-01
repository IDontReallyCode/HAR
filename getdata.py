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
    ticker = 'VOO'
    # ticker = 'TSLA'
    data = td.Get_Ticker_nMinuteSampling(ticker, nminutesampling=nminute)
    intradaydf = td.Get_TS_intraday_df(data)
    intradaydf.rename(columns={'close':'price', 'date_eod':'date'}, inplace=True)

    intradaydf.to_csv(f"./intradaysample{ticker}.csv")
    






#### __name__ MAIN()
if __name__ == '__main__':
    main()


