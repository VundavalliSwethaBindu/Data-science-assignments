# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 19:44:18 2023

@author: sweth
"""

# question 1: 
# Delivery_time -> Predict delivery time using sorting time

import pandas as pd
import numpy as np
df = pd.read_csv("delivery_time.csv")
df
df.iloc[:,0:3]

df.head()
# checking out for outliers
import matplotlib.pyplot as plt
plt.boxplot(df["Delivery Time"])
plt.show()
plt.boxplot(df["Sorting Time"])
plt.show()
#last five data elements of the data
df.tail()

df.isnull().sum()
# to check the rows and columns
df.shape
# describes all the necessary things like mean, standard deviation.......
df.describe()
#splitting X and Y where y needs to be predicted using X
x= df[["Sorting Time"]]
y=df["Delivery Time"]

# model fitting
from sklearn.linear_model import LinearRegression as Lr
Lr=Lr()
Lr.fit(x,y)
Ypred = Lr.predict(x)

# metrics
from sklearn.metrics import mean_squared_error
msqe=mean_squared_error(y,Ypred)
print("mean square error",msqe.round(2))
print("root mean square error",np.sqrt(msqe).round(2))

import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"],y=df["Delivery Time"],color='red')
plt.plot(df["Sorting Time"],Ypred,color='black')
plt.show()

a_new= int(input())
q = np.array([[a_new]])
y=Lr.predict(q)
b = y.round(2)

print("output for new x value",b)

# Adding the newly predicted value into original dataset:
c =pd.DataFrame({"Sorting Time":a_new,"Delivery Time":y[0]},index=[0])
df=df.append(c,ignore_index=True)
df

import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"],y=df["Delivery Time"],color='red')
plt.plot(df["Sorting Time"],Ypred,color="black")
plt.show()


# question 2
# Salary_hike -> Build a prediction model for Salary_hike
import pandas as pd
import numpy as np
df = pd.read_csv("Salary_data.csv")
df
df.iloc[:,0:3]

df.head()
# checking out for outliers
import matplotlib.pyplot as plt
plt.boxplot(df["YearsExperience"])
plt.show()
plt.boxplot(df["Salary"])
plt.show()
#last five data elements of the data
df.tail()

df.isnull().sum()
# to check the rows and columns
df.shape
# describes all the necessary things like mean, standard deviation.......
df.describe()
#splitting X and Y where y needs to be predicted using X
x= df[["YearsExperience"]]
y=df["Salary"]

# model fitting
from sklearn.linear_model import LinearRegression as Lr
Lr=Lr()
Lr.fit(x,y)
Ypred = Lr.predict(x)

# metrics
from sklearn.metrics import mean_squared_error
msqe=mean_squared_error(y,Ypred)
print("mean square error",msqe.round(2))
print("root mean square error",np.sqrt(msqe).round(2))

import matplotlib.pyplot as plt
plt.scatter(x=df["YearsExperience"],y=df["Salary"],color='red')
plt.plot(df["YearsExperience"],Ypred,color='black')
plt.show()

a_new= int(input())
q = np.array([[a_new]])
y=Lr.predict(q)
b = y.round(2)

print("output for new x value",b)

# Adding the newly predicted value into orginal dataset:
c =pd.DataFrame({"Sorting Time":a_new,"Delivery Time":y[0]},index=[0])
df=df.append(c,ignore_index=True)
df

import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"],y=df["Delivery Time"],color='red')
plt.plot(df["Sorting Time"],Ypred,color="black")
plt.show()





