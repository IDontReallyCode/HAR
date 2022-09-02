# HAR

My implementation of Heterogenous Auto-Regressive (HAR) model for forecasting Realised Volatility

This is developped as a set of simple functions to allow for better flexibility and control.
Convenient functions might be created to make the process easier at first.

The input should be a Dataframe of intraday data, with at least two columns
[price] that containts the closing price of the candle.
[date] that contains the date of the day (without time stamps).
[TODO] Allow other formats for the data.

With this library, you can:
- [x] Calculate the Realized Variance (using the simple definition)
- [ ] clean data and calculate a "better" Realized Variance
- [x] Aggregate the Realized Variance over multiple time horizon
- [ ] Possibility to add external factors to the regression
- [x] Estimate the coefficients for the basic HAR model using OLS or weighted Least-Squares
- [x] Estimate the coefficients for the HAR model with external factors
- [x] Use the CNGARCH packages to estimate CNGARCH and add to the forecast model
- [x] Use Random Forest Regression for the forecast over multiple time horizon
- [ ] HARQ with bias reduction for quarticity
- [x] use log-range instead of RV
- [ ] A function that deals with back-testing HAR forecast with a rolling-window and compares to Martingal forecast.



REFERENCES (I'll sort them later)

Clements, Adam, and Daniel PA Preve. "A practical guide to harnessing the har volatility model." Journal of Banking & Finance 133 (2021): 106285.
https://www.sciencedirect.com/science/article/pii/S0378426621002417


Corsi, Fulvio. "A simple approximate long-memory model of realized volatility." Journal of Financial Econometrics 7, no. 2 (2009): 174-196.
https://www.finanzaonline.com/forum/attachments/econometria-e-modelli-di-trading-operativo/1590979d1336906628-forecasting-realized-vol-punto-di-partenza-har-rv-2004-corsi-simple-long-memory-model-realized.pdf


Bollerslev, Tim, Andrew J. Patton, and Rogier Quaedvlieg. "Exploiting the errors: A simple approach for improved volatility forecasting." Journal of Econometrics 192, no. 1 (2016): 1-18.
https://www.sciencedirect.com/science/article/pii/S0304407615002584





