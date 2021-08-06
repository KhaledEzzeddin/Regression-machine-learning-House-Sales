 #_________________libraries__________
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
#__________________read the file____________
df=pd.read_csv("house.csv")
df.head()
#________________database info________________
df.info()
#__________finding the missing data _______________
df.isnull().sum()
#____________droping the unimportant data_____________________
df=df.drop(["id","date","lat","long"],axis=1)
df.head()
#________________ database___________________
sb.stripplot(x="yr_built",y="bedrooms",data=df)
#plt.show()
#__________________preparing the database to training the Algorithme ____________
columns=["bedrooms","bathrooms","sqft_living","sqft_lot","floors",
         "waterfront","view","condition","grade","sqft_above","sqft_basement"
         ,"yr_built","yr_renovated","zipcode","sqft_living15","sqft_lot15"]
labels=df["price"].values
features=df[list(columns)].values
print("\nthe labels is\n",labels)
print("\nthe features is\n",features)
#__________________________scaling_______________________
#__________________training and testing the Algorithme 1______________________
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.30)
#__________Regression the result ________________
regr=linear_model.LinearRegression()
regr.fit(x_train,y_train)
#__________________ score of training____________
accuracyTrain=regr.score(x_train,y_train)
print("\nthe accuracy Train is\n",accuracyTrain)
#____________score of testing______________
accuracyTest=regr.score(x_test,y_test)
print("\nthe accuracy of test is \n",accuracyTest)