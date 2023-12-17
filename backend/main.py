from fastapi import FastAPI
import yfinance as yf
from sklearn.linear_model import LinearRegression
import xgboost as ai
import pandas as pd
from typing import Union

app = FastAPI()

def SKLinReg(data):

  # Split the data into training and testing sets
  features= ['Open','Volume','High','Low']
  target = ['Close']
  traindata = data.iloc[:int(.90*len(data)),:]
  test = data.iloc[int(.90*len(data)):,:]

  # Create a linear regression model
  model = LinearRegression()

  # Fit the model to the training data
  model.fit(traindata[features], traindata['Close'])

  # Evaluate the model on the testing data
  score = model.score(test[features],  test['Close'])
  print('this model score is(sklearn) ',score)
  # Make predictions using the model
  pred = model.predict(test[features])
  return pred , score

def xgbpriceclose(data):
  features= ['Open','Volume','High','Low']
  traindata = data.iloc[:int(.90*len(data)),:]
  #data to test the model
  test = data.iloc[int(.90*len(data)):,:]
  model = ai.XGBRegressor()
  model.fit(traindata[features],traindata['Close'])
  predictions = model.predict(test[features])
  a=model.score(test[features],test['Close'])

  return predictions, a


def XgbSupLearn(data, days):
  #make model and set data so previous close price is input to find the next one
  model = ai.XGBRegressor()
  inputdata = data['Close'].shift(1)
  targetdata = data['Close']
  model.fit(inputdata, targetdata)
  pred = model.predict(inputdata.values[int(.90*len(data)):])
  future = list()
  score=model.score(inputdata,targetdata)

  for n in range(days):
      inputdata = data['Close'].shift(1)
      nextday = model.predict(inputdata[-1:])
      newday = inputdata.index[-1] + pd.Timedelta(days=1)
      newp = pd.DataFrame({'Date': [newday], 'Close': nextday})
      newp.set_index('Date', inplace=True)
      data = pd.concat([data,newp])
      future.append(newp)

  return pred,score,future
  #data grab
def stockdata(ticker,period):
  tickers = yf.Ticker(ticker)
  data = tickers.history(period=period)
  return data

def aimodels(ticker,period,days):
  data = stockdata(ticker,period)
  XGBgraph ,a = xgbpriceclose(data)
  SKLGraph, ska = SKLinReg(data)
  XGBSUPGraph ,score,pred = XgbSupLearn(data,days)
  scores = [a,ska,score]
  XGBgraph =XGBgraph.tolist()
  SKLGraph=SKLGraph.tolist()
  XGBSUPGraph.tolist()
  
  # Convert pred to a list of floats
  pred_floats = [df['Close'].values[0] for df in pred]

  return XGBgraph, SKLGraph, XGBSUPGraph, scores, pred_floats

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/{ticker}/{period}/{days}")
async def predict_stock(ticker: str, period: str, days: int):
    _, _, _, _, pred = aimodels(ticker, period, days)

    # Convert numpy.float32 to Python native float
    pred = [float(p) for p in pred]

    return {"prediction": pred}