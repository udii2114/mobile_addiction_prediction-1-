import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Mobile Addiction Prediction System")

age = st.number_input("Age",18,30)

gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

screen = st.selectbox(
    "Screen Time",
    ["Less than 2 hours","2–4","4–6","6–8","More than 8 hours"]
)

social = st.selectbox(
    "Social Media Usage",
    ["Less than 1 hour","1–2","2–4","4–6","More than 6 hours"]
)

gaming = st.selectbox(
    "Gaming / Entertainment Apps",
    ["0 hours","1–2","2–4","4–6","More than 6 hours"]
)

sleep = st.selectbox(
    "Sleep Time",
    ["Less than 5 hours","5–6","6–7","More than 7 hours"]
)

anxious = st.selectbox(
    "Anxious Without Phone",
    ["no","sometimes","yes"]
)

before = st.selectbox(
    "Use Phone Before Sleeping",
    ["Never","Sometimes","Always"]
)

if st.button("Predict"):

    data = pd.DataFrame(
        [[age,gender,screen,social,gaming,
          sleep,anxious,before]],

        columns=[
            "Age",
            "Gender",
            "screen time",
            "social media",
            "gaming or entertainment apps",
            "sleep time",
            "anxious without phone",
            "Use phone before sleeping"
        ]
    )

    for col in data.columns:
        if col != "Age":
            data[col] = encoders[col].transform(data[col])

    prediction = model.predict(data)

    result = encoders["addiction level"].inverse_transform(prediction)

    st.success(f"Prediction: {result[0]}")