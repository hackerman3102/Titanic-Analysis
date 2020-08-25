
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statistics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data

passengers=pd.read_csv('passengers.csv')
passengers['Sex'] = passengers.Sex.map({'female':0, 'male':1})
# Update sex column to numerical
mean=passengers['Age'].mean()
mean=round(mean)
# Update sex column to numerical
passengers['Age'].fillna(value=mean,inplace=True)
passengers['FirstClass']=passengers.Pclass.apply(lambda x:1 if x==1 else 0)
passengers['SecondClass']=passengers.Pclass.apply(lambda x:1 if x==2 else 0)
features=passengers[['Sex','Age','FirstClass','SecondClass']]
survival=passengers['Survived']
x_train,x_test,y_train,y_test=train_test_split(features,survival,test_size=0.2)
ss=StandardScaler()
ss.fit_transform(x_train)
ss.fit_transform(x_test)
model=LogisticRegression()
model.fit(x_train,y_train)
score=model.score(x_train,y_train)
print(score)
score2=model.score(x_test,y_test)
print(score2)
print(model.coef_)
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Yash = np.array([0.0,20.0,0.0,1.0])# Combine passenger arrays
sample_passengers=np.array([Jack,Rose,Yash])
# Scale the sample passenger features
ss.transform(sample_passengers)
print(sample_passengers)
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
# Make survival predictions!
