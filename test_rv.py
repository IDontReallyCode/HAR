import pandas as pd
from rv import rv
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import numpy as np

def main():
    data = pd.read_csv("./intradaysample.csv", index_col=0)
    
    # the realized variance requires 2 columns with specific names ['date'] and ['price']
    # ['date'] needs to be just a date. No time.
    data.rename(columns={'nicedate':'date', 'close':'price'}, inplace=True)

    rvdays, realizeddailyvariance = rv.rv(data)

    multiplesampling = rv.rvaggregate(realizeddailyvariance)

    dates = pltd.date2num(rvdays)
    # plt.plot_date(dates, np.sqrt(realizeddailyvariance*252), 'b-')
    # plt.show()

    plt.plot_date(dates, np.sqrt(multiplesampling[:,0]*252), 'b-', label='daily')
    plt.plot_date(dates, np.sqrt(multiplesampling[:,1]*252), 'g-', label='weekly')
    plt.plot_date(dates, np.sqrt(multiplesampling[:,2]*252), 'r-', label='bi-weekly')
    plt.plot_date(dates, np.sqrt(multiplesampling[:,3]*252), 'k-', label='monthly')
    plt.legend()
    plt.show()



#### __name__ MAIN()
if __name__ == '__main__':
    main()


