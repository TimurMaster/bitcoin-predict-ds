import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot
df = pd.read_csv('BTC-USD.csv')
df = df[["Date", "Close"]]
df.columns = ["ds", "y"]

prophet = Prophet()
prophet.fit(df)
future = prophet.make_future_dataframe(periods=365)

forecast = prophet.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(200)
prophet.plot(forecast, figsize=(20, 10))
