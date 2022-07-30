# HAR

My implementation of Heterogenous Auto-Regressive (HAR) model for forecasting Realised Volatility

(citations to be added eventually)

The input should be a Dataframe of intraday data, with at least two columns
[price] that containts the closing price of the candle.
[date] that contains the date of the day (without time stamps).

With this library, you can:
- Calculate the Realized Variance (using the simple definition)
- [TODO] clean data and calculate a "better" Realized Variance
- Aggregate the Realized Variance over multiple time horizon
- [TODO] Possibility to add external factors to the regression
- Estimate the coefficients for the basic HAR model
- Estimate the coefficients for the HAR model with external factors
- Use the CNGARCH packages to estimate CNGARCH and add to the forecast model
- Use Random Forest Regression for the forecast over multiple time horizon

