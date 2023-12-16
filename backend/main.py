from fastapi import FastAPI
import yfinance as yf
from sklearn.linear_model import LinearRegression
import xgboost as ai
import pandas as pd
from typing import Union

app = FastAPI()

# get prediction
def xgbpriceclose(data):
  features= ['Open','Volume','High','Low']
  traindata = data.iloc[:int(.90*len(data)),:]
  #data to test the model
  test = data.iloc[int(.90*len(data)):,:]
  model = ai.XGBRegressor()
  model.fit(traindata[features],traindata['Close'])
  predictions = model.predict(test[features])
  print(predictions)
  a=model.score(test[features],test['Close'])
  print('this model score is(XGB) ',a)
  return predictions, a

#data grab
def stockdata(ticker,period):
  tickers = yf.Ticker(ticker)
  data = tickers.history(period=period)
  return data


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/{ticker}/{period}")
async def predict_stock(ticker: str, period: str):
    ticker = 'mcd'
    period = '1y'
    data = stockdata(ticker,period)

    XgbP ,a =xgbpriceclose(data)

    lastest_prediction = float(list(XgbP)[-1])

    return {"ticker": ticker, "period": period, "prediction": lastest_prediction}
