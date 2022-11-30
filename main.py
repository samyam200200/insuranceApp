# Insurance price prediction using Random Forest Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("insurance.csv")

# Encoding categorical data using LabelEncoder
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
dataset["sex"] = labelencoder.fit_transform(dataset["sex"])
dataset["smoker"] = labelencoder.fit_transform(dataset["smoker"])
dataset["region"] = labelencoder.fit_transform(dataset["region"])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Evaluating the Model Performance
from sklearn import metrics

print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 score:", metrics.r2_score(y_test, y_pred))

# Save the model
import pickle

pickle.dump(regressor, open("model.pkl", "wb"))
