import streamlit as st
import joblib
import numpy as np

model = joblib.load('linear_regression_model.pkl')

st.title('Linear Predictor (13 Features Required)')

CRIM = st.slider('CRIM', min_value=0.0, max_value=100.0, value=0.0)
ZN = st.slider('ZN', min_value=0.0, max_value=100.0, value=0.0)
INDUS = st.slider('INDUS', min_value=0.0, max_value=30.0, value=0.0)
CHAS = st.slider('CHAS', min_value=0, max_value=1, value=0)
NOX = st.slider('NOX', min_value=0.3, max_value=0.9, value=0.3)
RM = st.slider('RM', min_value=3.5, max_value=9.5, value=6.0)
AGE = st.slider('AGE', min_value=0.0, max_value=100.0, value=50.0)
DIS = st.slider('DIS', min_value=1.0, max_value=12.0, value=5.0)
RAD = st.slider('RAD', min_value=1, max_value=24, value=1)
TAX = st.slider('TAX', min_value=100, max_value=700, value=300)
PTRATIO = st.slider('PTRATIO', min_value=12.0, max_value=22.0, value=18.0)
B = st.slider('B', min_value=0.0, max_value=400.0, value=100.0)
LSTAT = st.slider('LSTAT', min_value=1.0, max_value=40.0, value=10.0)

# Collect features into a list
features = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]

# Prediction
if st.button('Predict'):
    prediction = model.predict([features])[0]
    st.success('Predicted Value: ' + str(prediction))
