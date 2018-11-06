import pandas as pd
import quandl
import math,datetime
import numpy as np 
from sklearn import preprocessing, svm 
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style

import pickle

style.use('ggplot')

df=quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
# print(df.head())
df['ML_PCT']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Close']*100.0
df['PCT_Change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

df=df[['Adj. Close','ML_PCT','PCT_Change','Adj. Volume']]
# print(df.head())
forecast_col= 'Adj. Close'
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
# print(df.head())

#Features represented as Uppercase 'X'
#Lbels represenrteed as LowerCase 'y'
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X=X[:-forecast_out]
X_lately = X[-forecast_out:]
# X = X[:-forecast_out+1]
# df.dropna(inplace=True)
df.dropna(inplace=True)
y = np.array(df['label'])

# print(len(X),len(y))
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

classifier = LinearRegression(n_jobs=-1)
classifier.fit(X_train,y_train)

with open('linearregresseion.pickle','wb') as f:
    pickle.dump(classifier,f)

pickle_in = open('linearregresseion.pickle','rb')
classifier = pickle.load(pickle_in)
accuracy = classifier.score(X_test,y_test)

# print ("Acccuracy:",accuracy)
forecast_set = classifier.predict(X_lately)
print(forecast_set,accuracy,forecast_out)

df['Forecast']  = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day  = 86400
next_unix = last_unix+one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('Price')
plt.show()