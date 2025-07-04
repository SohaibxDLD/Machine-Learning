import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")

def load_data():
    dataset = pd.read_csv('boston.csv')
    return dataset

def remove_outliers(dataset):
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset[(dataset >= (Q1 - 1.5 * IQR)) & (dataset <= (Q3 + 1.5 * IQR))]
    return dataset.dropna()

def main():
    st.title("🏠 Boston Housing Price Analysis")
    
    dataset = load_data()
    dataset = remove_outliers(dataset)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset", "Statistics", "Distributions", "Relationships", "Prediction"])
    
    with tab1:
        st.header("Raw Dataset")
        st.dataframe(dataset, height=400)
        
        st.subheader("Dataset Information")
        st.text("Shape: " + str(dataset.shape))
        
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Missing Values")
        st.dataframe(dataset.isnull().sum().to_frame("Missing Values"))
    
    with tab2:
        st.header("Descriptive Statistics")
        st.dataframe(dataset.describe())
        
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    
    with tab3:
        st.header("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Histograms")
            selected_feature = st.selectbox("Select feature for histogram", dataset.columns)
            fig, ax = plt.subplots()
            sns.histplot(dataset[selected_feature], kde=True, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Box Plots")
            selected_feature_box = st.selectbox("Select feature for box plot", dataset.columns)
            fig, ax = plt.subplots()
            sns.boxplot(data=dataset, y=selected_feature_box, ax=ax)
            st.pyplot(fig)
    
    with tab4:
        st.header("Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Scatter Plots")
            x_feature = st.selectbox("X-axis feature", dataset.columns.drop('MEDV'))
            y_feature = st.selectbox("Y-axis feature", ['MEDV'] + list(dataset.columns.drop('MEDV')))
            fig, ax = plt.subplots()
            sns.scatterplot(data=dataset, x=x_feature, y=y_feature, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Pair Plot (Sample)")
            sample_size = st.slider("Sample size", 10, 100, 50)
            features = st.multiselect("Select features", dataset.columns, default=['RM', 'LSTAT', 'PTRATIO', 'MEDV'])
            fig = sns.pairplot(dataset[features].sample(sample_size))
            st.pyplot(fig)
    
    with tab5:
        st.header("Price Prediction")
        
        model = joblib.load('linear_regression_model.pkl')
        scaler = joblib.load('scaler1.pkl')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Enter Feature Values")
            CRIM = st.number_input('Crime rate (CRIM)', min_value=0.0, max_value=100.0, value=0.0)
            ZN = st.number_input('Residential land (ZN)', min_value=0.0, max_value=100.0, value=0.0)
            INDUS = st.number_input('Non-retail business acres (INDUS)', min_value=0.0, max_value=30.0, value=0.0)
            CHAS = st.selectbox('Charles River dummy (CHAS)', [0, 1])
            NOX = st.number_input('Nitric oxides (NOX)', min_value=0.3, max_value=0.9, value=0.5)
            RM = st.number_input('Avg rooms (RM)', min_value=3.0, max_value=10.0, value=6.0)
            AGE = st.number_input('Old homes (AGE)', min_value=0.0, max_value=100.0, value=50.0)
        
        with col2:
            st.subheader("")
            DIS = st.number_input('Distance to employment (DIS)', min_value=1.0, max_value=12.0, value=5.0)
            RAD = st.number_input('Highway accessibility (RAD)', min_value=1, max_value=24, value=5)
            TAX = st.number_input('Tax rate (TAX)', min_value=100, max_value=700, value=300)
            PTRATIO = st.number_input('Pupil-teacher ratio (PTRATIO)', min_value=12.0, max_value=22.0, value=18.0)
            B = st.number_input('Black population (B)', min_value=0.0, max_value=400.0, value=350.0)
            LSTAT = st.number_input('Lower status % (LSTAT)', min_value=1.0, max_value=40.0, value=10.0)
        
        if st.button('Predict Price'):
            features = np.array([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            st.success(f'Predicted Home Value: ${prediction[0]*1000:,.2f}')
            
            st.subheader("Residual Analysis")
            y_test = dataset['MEDV']
            X_test = dataset.drop('MEDV', axis=1)
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            
            sns.scatterplot(x=y_pred, y=y_test-y_pred, ax=ax[0])
            ax[0].axhline(y=0, color='r', linestyle='--')
            ax[0].set_title('Residuals vs Predicted')
            ax[0].set_xlabel('Predicted Values')
            ax[0].set_ylabel('Residuals')
            
            sns.histplot(y_test-y_pred, kde=True, ax=ax[1])
            ax[1].set_title('Residuals Distribution')
            
            st.pyplot(fig)
            
            st.subheader("Model Performance")
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            st.metric("Mean Squared Error", f"{mse:.2f}")
            st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            st.metric("R2 Score", f"{r2*100:.2f}%")

if __name__ == "__main__":
    import io
    main()