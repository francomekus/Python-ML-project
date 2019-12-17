# WE WANT TO USE MACHINE LEARNING TO DETERMINE WHICH
# PHYSIOCHEMICAL PROPERTIES MAKE A GOOD WINE
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
wine_data=pd.read_csv(r"C:\Users\asus\Desktop\winequality-red.csv")
#https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
# checking for duplicate.
wine_data.describe()
# our input dataset
X = wine_data.drop(columns=["quality"])
# our output dataset
y = wine_data["quality"]
# i am using DecisionTreeClassifer Algorithm
model = DecisionTreeClassifier()
model.fit(X,y)
x_in=np.array([6.0,0.310,0.47,3.6,0.067,18.0,42.0,0.99549,3.39,0.66,11.0])
predictions = model.predict([x_in])
print(predictions)