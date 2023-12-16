from fastapi import FastAPI
import yfinance as yf
from sklearn.linear_model import LinearRegression
import xgboost as ai
import pandas as pd
from typing import Union



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
  return predictions[-1]


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

ticker = 'mcd'
period = '1y'
data = stockdata(ticker,period)
print(data)

XgbP ,a =xgbpriceclose(data)
print(XgbP)