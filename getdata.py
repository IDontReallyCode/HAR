from tdalocal import TDA_STUFF as td

def main():
    # get n minute data for MSFT
    nminute = 5
    ticker = 'MSFT'
    mindata = td.Get_Ticker_nMinuteSampling(ticker, nminutesampling=nminute)
    

    print(mindata)
    






#### __name__ MAIN()
if __name__ == '__main__':
    main()


