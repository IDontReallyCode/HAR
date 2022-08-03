import pandas as pd
from HAR import rv
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import numpy as np
import time

def main():
    ticker = 'SPY'
    # data = pd.read_csv(f"./intradaysample{ticker}.csv", index_col=0)
    data = pd.read_csv(f"./{ticker}_5m.csv", index_col=0)

    # the realized variance requires 2 columns with specific names ['date'] and ['price']
    # ['date'] needs to be just a date. No time.
    # data.rename(columns={'nicedate':'date', 'close':'price'}, inplace=True)

    s = time.time()
    realizeddailyvariance, rvdays = rv.rv(data, datecolumnname='nicedate', closingpricecolumnname='close')
    print(time.time()-s)

    datesofalldays = np.array(data['nicedate'])
    intradayclosingprices = np.array(data['close'])

    # s = time.time()
    # numpyrv = rv.rvnumpy(datesofalldays, intradayclosingprices)
    # print(time.time()-s)

    # s = time.time()
    # numpyrv = rv.rvscipy(datesofalldays, intradayclosingprices)
    # print(time.time()-s)

    done=1




#### __name__ MAIN()
if __name__ == '__main__':
    main()


