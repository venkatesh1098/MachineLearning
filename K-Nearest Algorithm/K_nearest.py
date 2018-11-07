import numpy 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_val_score,train_test_split
import pandas as pd 

df = pd.read_csv('/home/venkatesh/Desktop/Machine Learning/K-Nearest Algorithm/breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace = True)
#X = features y= Labels
X = numpy.array(df.drop(['class'],1))
y = numpy.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(accuracy)


example_measures = numpy.array([[4,2,1,1,1,2,3,2,1],[4,1,3,2,3,2,4,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)#number of datta to predict is used in place of 2 sisnce we are using two dataas for prediction

prediction = classifier.predict(example_measures)
print(prediction)