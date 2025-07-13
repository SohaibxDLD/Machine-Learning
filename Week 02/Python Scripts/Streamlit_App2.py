import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "http://localhost:8000" 
st.set_page_config(page_title="Wine Classification", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Wine.csv")
    df = df.drop(['Ash', 'Mg', 'Color.int'], axis=1)
    return df

def get_available_models():
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()["available_models"]
        return []
    except requests.exceptions.RequestException:
        return []

def main():
    st.title("🍷 Wine Classification")
    
    tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Info"])
    
    with tab1:
        st.header("Make a Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            alcohol = st.slider("Alcohol", 11.0, 15.0, 13.0)
            malic_acid = st.slider("Malic Acid", 0.5, 6.0, 2.0)
            alcalinity = st.slider("Alcalinity", 10.0, 30.0, 20.0)
            phenols = st.slider("Phenols", 0.5, 4.0, 2.0)
            flavanoids = st.slider("Flavanoids", 0.5, 6.0, 2.0)
        
        with col2:
            nonflavanoid = st.slider("Nonflavanoid Phenols", 0.1, 1.0, 0.3)
            proanthocyanins = st.slider("Proanthocyanins", 0.5, 4.0, 2.0)
            hue = st.slider("Hue", 0.5, 2.0, 1.0)
            od = st.slider("OD280/OD315", 1.0, 4.0, 2.0)
            proline = st.slider("Proline", 200.0, 1700.0, 1000.0)
            model_type = st.selectbox("Model", get_available_models(), key="model_type")
        
        if st.button("Predict Wine Class", key="predict_button"):
            try:
                wine_data = {
                    "Alcohol": alcohol,
                    "Malic_acid": malic_acid,
                    "Alcalinity": alcalinity,
                    "Phenols": phenols,
                    "Flavanoids": flavanoids,
                    "Nonflavanoid_phenols": nonflavanoid,
                    "Proanthocyanins": proanthocyanins,
                    "Hue": hue,
                    "OD280_OD315": od,
                    "Proline": proline,
                    "model_type": model_type
                }
                
                response = requests.post(f"{API_URL}/predict", json=wine_data)
                
                if response.status_code == 200:
                    result = response.json()
                    wine_class = result["wine_class"]
                    colors = ['red', 'green', 'blue']
                    
                    st.markdown(f"<h2 style='color:{colors[wine_class-1]};'>Predicted Class: {wine_class}</h2>", 
                               unsafe_allow_html=True)
                    
                    if result["probability"]:
                        probs = result["probability"][0]
                        fig, ax = plt.subplots()
                        ax.bar(['Class 1', 'Class 2', 'Class 3'], probs, color=colors)
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Confidence')
                        st.pyplot(fig)
                else:
                    st.error(f"Error: {response.text}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the API: {e}")
    
    with tab2:
        st.header("Data Analysis")
        df = load_data()
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        st.subheader("Statistics")
        st.dataframe(df.describe())
        
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Distribution by Feature")
        feature = st.selectbox("Select feature", df.columns[:-1], key="feature_select")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Wine', y=feature, data=df, ax=ax2)
        st.pyplot(fig2)
    
    with tab3:
        st.header("Model Information")
        st.markdown("""
        ### Available Models:
        - **Decision Tree**: A tree-like model that makes decisions based on feature thresholds
        - **KNN (K-Nearest Neighbors)**: Classifies based on similarity to training examples
        - **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
        
        ### Features Used:
        - Alcohol
        - Malic Acid
        - Alcalinity of Ash  
        - Phenols
        - Flavanoids
        - Nonflavanoid Phenols
        - Proanthocyanins
        - Color Intensity (Hue)
        - OD280/OD315
        - Proline
        """)

if __name__ == "__main__":
    main()