"""
Do not use this program.
Get your own intraday data using your available means.
This program is offered as an example only
"""
from tdalocal import TDA_STUFF as td
import pandas as pd

def main():
    # get "n" minute data for "ticker"
    nminute = 10
    ticker = 'CLF'
    # ticker = 'TSLA'
    data = td.Get_Ticker_nMinuteSampling(ticker, nminutesampling=nminute)
    intradaydf = td.Get_TS_intraday_df(data)
    # intradaydf.rename(columns={'date_eod':'date'}, inplace=True)

    intradaydf.to_csv(f"./intradaysample{ticker}.csv")
    """
    example of data
    intradaydf
           date        timestamp            price    volume
    0      2021-11-15  2021-11-15 03:05:00  430.500  500
    1      2021-11-15  2021-11-15 03:10:00  430.500  100
    2      2021-11-15  2021-11-15 05:25:00  430.600  100
    3      2021-11-15  2021-11-15 06:30:00  430.570  547
    4      2021-11-15  2021-11-15 06:40:00  430.600  100

    """

    # get daily data for same ticker

    dailydf = td.Get_Ticker_Daily_df(ticker)

    dailydf.to_csv(f"./dailysample{ticker}.csv")

    """
    Example
    dailydf
          date_eod    close    high    low        volume
    0     2010-09-09  101.320  102.50  101.1400    26513
    1     2010-09-10  101.780  101.86  101.2960     8638
    2     2010-09-13  103.060  103.14  102.5000    33752
    3     2010-09-14  103.038  103.48  102.3800    59420
    4     2010-09-15  103.300  103.38  102.4000     9283
    """

    done=1

    






#### __name__ MAIN()
if __name__ == '__main__':
    main()


