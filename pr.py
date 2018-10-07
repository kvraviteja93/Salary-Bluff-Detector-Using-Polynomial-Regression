#Importing The Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing The Dataset
dataset=pd.read_csv("Position_Salaries.csv")

#Separating Independent and Dependent Variables
X=dataset.iloc[:,[1]].values
y=dataset.iloc[:,-1].values

#Splitting The Dataset into Training and Test Set
'''from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,split_ratio=0.2)'''

#Fitting Linear Regression To The Dataset
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,y)

#Fitting Polynomial Regression To The Dataset
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)
poly.fit(X,y)
polyreg=LinearRegression()
polyreg.fit(X_poly,y)

#Visualizing The Linear Regression Results
plt.scatter(X,y,color='red')
plt.plot(X,linreg.predict(X),color='green')
plt.xlabel("Position")
plt.ylabel("Salaries")
plt.title("Salary Bluff Detector Using Linear Regression")
plt.show()

#Visualizing The Polynomial Regression Results
plt.scatter(X,y,color='red')
plt.plot(X,polyreg.predict(poly.fit_transform(X)),color='blue')
plt.xlabel("Position")
plt.ylabel("Salaries")
plt.title("Salary Bluff Detector Using Polynomial Regression")
plt.show()

#Predicting Individual Position Salaries
linreg.predict(6.5)
polyreg.predict(poly.fit_transform(6.5))