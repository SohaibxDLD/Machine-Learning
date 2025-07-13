import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "http://localhost:8000" 
st.set_page_config(page_title="Titanic Survival", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("titanic.csv")
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].copy()
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
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
    st.title("🚢 Titanic Survival Prediction")
    
    tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Info"])
    
    with tab1:
        st.header("Make a Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], key="pclass")
            sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
            age = st.slider("Age", 0, 100, 30, key="age")
        
        with col2:
            fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, key="fare")
            model_type = st.selectbox("Model", get_available_models(), key="model_type")
        
        if st.button("Predict Survival", key="predict_button"):
            try:
                passenger_data = {
                    "Pclass": pclass,
                    "Sex": sex.lower(),
                    "Age": age,
                    "Fare": fare,
                    "model_type": model_type
                }
                
                response = requests.post(f"{API_URL}/predict", json=passenger_data)
                
                if response.status_code == 200:
                    result = response.json()
                    survival = "Survived" if result["survived"] else "Did Not Survive"
                    color = "green" if result["survived"] else "red"
                    
                    st.markdown(f"<h2 style='color:{color};'>Prediction: {survival}</h2>", 
                               unsafe_allow_html=True)
                    
                    if result["probability"]:
                        probs = result["probability"][0]
                        fig, ax = plt.subplots()
                        ax.bar(['Did Not Survive', 'Survived'], probs, color=['red', 'green'])
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
        
        st.subheader("Survival by Feature")
        feature = st.selectbox("Select feature", ['Pclass', 'Sex', 'Age', 'Fare'], key="feature_select")
        
        if feature == 'Sex':
            fig2, ax2 = plt.subplots()
            sns.barplot(x='Sex', y='Survived', data=df, ax=ax2)
            ax2.set_xticklabels(['Female', 'Male'])
            st.pyplot(fig2)
        else:
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='Survived', y=feature, data=df, ax=ax2)
            st.pyplot(fig2)
    
    with tab3:
        st.header("Model Information")
        st.markdown("""
        ### Available Models:
        - **Decision Tree**: A tree-like model that makes decisions based on feature thresholds
        - **KNN (K-Nearest Neighbors)**: Classifies based on similarity to training examples
        - **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
        
        ### Features Used:
        - Passenger Class (1st, 2nd, 3rd)
        - Sex (Male/Female)
        - Age
        - Fare paid
        """)

if __name__ == "__main__":
    main()