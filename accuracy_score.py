import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
wine_data=pd.read_csv(r"C:\Users\asus\Desktop\winequality-red.csv")
# our input dataset
X = wine_data.drop(columns=["quality"])
# our output dataset
y = wine_data["quality"]
# we are splitting our dataset allocating 20% for testing
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
score = accuracy_score(y_test, predictions)
print(score)
