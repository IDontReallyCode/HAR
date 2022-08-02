import pandas as pd
from HAR import rv
from HAR import HAR
import matplotlib.pyplot as plt
import numpy as np

def main():
    ticker = 'SPY'
    # data = pd.read_csv(f"./intradaysample{ticker}.csv", index_col=0)
    data = pd.read_csv(f"./{ticker}_5m.csv", index_col=0)
    
    # the realized variance requires 2 columns with specific names ['date'] and ['price']
    # ['date'] needs to be just a date. No time.
    # data.rename(columns={'nicedate':'date', 'close':'price'}, inplace=True)
    aggregatesampling = [1,5,10,22,63]

    HARresults = HAR.estimateforecast(data, aggregatesampling=aggregatesampling, datecolumnname='nicedate', closingpricecolumnname='close')

    done=1


#### __name__ MAIN()
if __name__ == '__main__':
    main()


