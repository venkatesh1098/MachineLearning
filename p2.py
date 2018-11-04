import pandas as pd
import quandl
import math
import numpy as np 
from sklearn import preprocessing, svm 
from sklearn.linear_model import LinearRegression


df=quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
# print(df.head())
df['ML_PCT']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Close']*100.0
df['PCT_Change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

df=df[['Adj. Close','ML_PCT','PCT_Change','Adj. Volume']]
# print(df.head())
forecast_col= 'Adj. Close'
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
# print(df.head())

#Features represented as Uppercase 'X'
#Lbels represenrteed as LowerCase 'y'
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)

X = X[:-forecast_out+1]
df.dropna(inplace=True)
y = np.array(df['label'])

# print(len(X),len(y))
