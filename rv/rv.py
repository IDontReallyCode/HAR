import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates

def rv(data:pd.DataFrame):
    # This function requires a dataframe with two columns ['date'] and ['price']
    # ['date'] needs to be just a date. No time.

    data['lr2'] = (np.log(data.price) - np.log(data.price.shift(1)))**2
    data = data[data['lr2'].notna()]

    alldays = data['date'].unique()
    nbdays = len(alldays)
    realizeddailyvariance = np.zeros((nbdays,))

    idx=0
    for day, g in data.groupby('date'):
        realizeddailyvariance[idx] = sum(g['lr2'])
        if np.sqrt(realizeddailyvariance[idx]*252)<0.1:
            print(g['date'].iloc[0])
        idx+=1

    return alldays, realizeddailyvariance

