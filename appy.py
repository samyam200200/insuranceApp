# Creating a FastAPI for prediction of insurance price
# Path: app.py
# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model
model = pickle.load(open("model.pkl", "rb"))

# Create the app object
app = FastAPI()

# Create a class for the request body
class RequestBody(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


# Create a class for the response body
class ResponseBody(BaseModel):
    prediction: float


# Create the predict endpoint
@app.post("/predict", response_model=ResponseBody)
def predict(request_body: RequestBody):
    # Get the request body data
    age = request_body.age
    sex = request_body.sex
    bmi = request_body.bmi
    children = request_body.children
    smoker = request_body.smoker
    region = request_body.region

    # Encode the categorical data
    # first we need to load the dataset
    dataset = pd.read_csv("insurance.csv")
    # then we need to encode the categorical data
    from sklearn.preprocessing import LabelEncoder

    labelencoder = LabelEncoder()
    labelencoder.fit(dataset["sex"])
    sex = labelencoder.transform([sex])
    labelencoder.fit(dataset["smoker"])
    smoker = labelencoder.transform([smoker])
    labelencoder.fit(dataset["region"])
    region = labelencoder.transform([region])

    # Combine the values into a single array
    print("_______")
    data = np.array([age, sex, bmi, children, smoker, region])

    # Reshape the array into a single row
    data = data.reshape(1, -1)

    # Make the prediction
    prediction = model.predict(data)
    return {"prediction": prediction[0]}
