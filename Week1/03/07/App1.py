import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = joblib.load('logistic_regression_model.pkl')
encoder = joblib.load('iris_encoder.pkl')

st.title('🌸 Iris Flower Classifier')
st.write('Predict Iris species from measurements')

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 5.0, 3.0)
with col2:
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 3.8)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.2)

if st.button('Predict'):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    species = encoder.inverse_transform(prediction)[0]
    
    st.success(f'Predicted Species: {species}')
    
    proba = model.predict_proba(input_data)[0]
    st.write('Prediction probabilities:')
    for i, (class_, prob) in enumerate(zip(encoder.classes_, proba)):
        st.write('class', i+1, ':', class_, 'probability:', prob * 100, '%')