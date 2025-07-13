import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Breast Cancer Diagnosis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("breast-cancer.csv")
    df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    return df

def get_available_models():
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json().get("available_models", [])
        return []
    except requests.exceptions.RequestException:
        return ["decision_tree", "knn", "naive_bayes"]

def main():
    st.title("🩺 Breast Cancer Diagnosis Predictor")
    
    if 'available_models' not in st.session_state:
        st.session_state.available_models = get_available_models()
    
    tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Info"])
    
    with tab1:
        st.header("Make a Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            # Mean features
            radius_mean = st.slider("Radius Mean", 6.0, 30.0, 14.0)
            texture_mean = st.slider("Texture Mean", 9.0, 40.0, 19.0)
            perimeter_mean = st.slider("Perimeter Mean", 43.0, 190.0, 92.0)
            area_mean = st.slider("Area Mean", 143.0, 2500.0, 654.0)
            smoothness_mean = st.slider("Smoothness Mean", 0.05, 0.17, 0.1)
            compactness_mean = st.slider("Compactness Mean", 0.02, 0.35, 0.1)
            concavity_mean = st.slider("Concavity Mean", 0.0, 0.43, 0.1)
            concave_points_mean = st.slider("Concave Points Mean", 0.0, 0.2, 0.05)
            symmetry_mean = st.slider("Symmetry Mean", 0.1, 0.3, 0.2)
            fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.05, 0.1, 0.06)
        
        with col2:
            # SE features (excluding fractal_dimension_se to match 19 features)
            radius_se = st.slider("Radius SE", 0.1, 2.9, 0.4)
            texture_se = st.slider("Texture SE", 0.2, 4.9, 1.2)
            perimeter_se = st.slider("Perimeter SE", 0.7, 22.0, 2.9)
            area_se = st.slider("Area SE", 6.0, 550.0, 40.0)
            smoothness_se = st.slider("Smoothness SE", 0.001, 0.03, 0.007)
            compactness_se = st.slider("Compactness SE", 0.002, 0.14, 0.025)
            concavity_se = st.slider("Concavity SE", 0.0, 0.4, 0.03)
            concave_points_se = st.slider("Concave Points SE", 0.0, 0.05, 0.01)
            symmetry_se = st.slider("Symmetry SE", 0.008, 0.08, 0.02)
            model_type = st.selectbox("Model", st.session_state.available_models)
        
        if st.button("Predict Diagnosis"):
            try:
                cancer_data = {
                    "radius_mean": radius_mean,
                    "texture_mean": texture_mean,
                    "perimeter_mean": perimeter_mean,
                    "area_mean": area_mean,
                    "smoothness_mean": smoothness_mean,
                    "compactness_mean": compactness_mean,
                    "concavity_mean": concavity_mean,
                    "concave_points_mean": concave_points_mean,
                    "symmetry_mean": symmetry_mean,
                    "fractal_dimension_mean": fractal_dimension_mean,
                    "radius_se": radius_se,
                    "texture_se": texture_se,
                    "perimeter_se": perimeter_se,
                    "area_se": area_se,
                    "smoothness_se": smoothness_se,
                    "compactness_se": compactness_se,
                    "concavity_se": concavity_se,
                    "concave_points_se": concave_points_se,
                    "symmetry_se": symmetry_se,
                    # Omit fractal_dimension_se to match 19 features
                    "model_type": model_type
                }
                
                response = requests.post(f"{API_URL}/predict", json=cancer_data, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    diagnosis = result["diagnosis"]
                    color = "red" if diagnosis == "Malignant" else "green"
                    st.markdown(f"<h2 style='color:{color};'>Prediction: {diagnosis}</h2>", unsafe_allow_html=True)
                    
                    if result.get("probability"):
                        probs = result["probability"][0]
                        fig, ax = plt.subplots()
                        ax.bar(['Benign', 'Malignant'], probs, color=['green', 'red'])
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Confidence')
                        st.pyplot(fig)
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Connection failed: {str(e)}")
    
    with tab2:
        st.header("Data Analysis")
        df = load_data()
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        st.subheader("Statistics")
        st.dataframe(df.describe())
        
        st.subheader("Correlation Matrix")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Distribution by Diagnosis")
        feature = st.selectbox("Select feature", numeric_df.columns[:-1], key="feature_select")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='diagnosis', y=feature, data=df, ax=ax2)
        ax2.set_xticklabels(['Benign', 'Malignant'])
        st.pyplot(fig2)
    
    with tab3:
        st.header("Model Information")
        st.markdown("""
        ### Available Models:
        - **Decision Tree**: A tree-like model that makes decisions based on feature thresholds
        - **KNN (K-Nearest Neighbors)**: Classifies based on similarity to training examples
        - **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
        
        ### Features Used (19 total):
        - Mean values of:
          - Radius, Texture, Perimeter, Area
          - Smoothness, Compactness, Concavity
          - Concave points, Symmetry, Fractal dimension
        - Standard Error of:
          - Radius, Texture, Perimeter, Area
          - Smoothness, Compactness, Concavity
          - Concave points, Symmetry
        """)

if __name__ == "__main__":
    main()