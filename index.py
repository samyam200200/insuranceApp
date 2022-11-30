import streamlit as st
import json
import requests

st.title("Insurance Price Prediction")

# Hit the API
url = "http://127.0.0.1:8000/predict"

# Get the user input
age = st.number_input("Age", min_value=18, max_value=100, value=18)
sex = st.selectbox("Sex", ("Male", "Female"))
bmi = st.number_input("BMI", min_value=15.96, max_value=53.13, value=25.0)
children = st.number_input("Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", ("Yes", "No"))
region = st.selectbox("Region", ("Northeast", "Northwest", "Southeast", "Southwest"))

if st.button("Calculate"):
    # Create the request body
    request_body = {
        "age": age,
        "sex": sex.lower(),
        "bmi": bmi,
        "children": children,
        "smoker": smoker.lower(),
        "region": region.lower(),
    }
    res = requests.post(url, data=json.dumps(request_body))
    st.write(res.json()["prediction"])
