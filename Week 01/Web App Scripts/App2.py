import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
import joblib
import streamlit as st

st.set_page_config(page_title="Iris Classification", layout="wide")

def load_data(filepath='Iris.csv'):
    dataset = pd.read_csv(filepath)
    dataset.drop('Id', axis=1, inplace=True)
    dataset.drop_duplicates(inplace=True)
    return dataset

page = st.sidebar.radio("Navigation", ["Data Exploration", "Model Training", "Prediction"])

dataset = load_data()

dataset_encoded = dataset.copy()
le = LabelEncoder()
dataset_encoded['Species'] = le.fit_transform(dataset_encoded['Species'])

if page == "Data Exploration":
    st.title("Iris Dataset Exploration")
    
    st.header("Dataset Preview")
    st.dataframe(dataset.head())
    
    st.header("Basic Statistics")
    st.dataframe(dataset.describe())
    
    st.header("Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Histograms", "Scatter Plots", "Correlation"])
    
    with tab1:
        st.subheader("Feature Distributions")
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        for i, feature in enumerate(features):
            row, col = i//2, i%2
            dataset[feature].hist(ax=ax[row, col])
            ax[row, col].set_title(feature)
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Scatter Plots")
        color_map = {'Iris-setosa': 'red', 'Iris-versicolor': 'black', 'Iris-virginica': 'blue'}
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Sepal Length vs Width")
            fig, ax = plt.subplots()
            for species, color in color_map.items():
                subset = dataset[dataset['Species'] == species]
                ax.scatter(subset['SepalLengthCm'], subset['SepalWidthCm'], 
                          color=color, label=species)
            ax.set_xlabel('Sepal Length')
            ax.set_ylabel('Sepal Width')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            st.write("Petal Length vs Width")
            fig, ax = plt.subplots()
            for species, color in color_map.items():
                subset = dataset[dataset['Species'] == species]
                ax.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'], 
                          color=color, label=species)
            ax.set_xlabel('Petal Length')
            ax.set_ylabel('Petal Width')
            ax.legend()
            st.pyplot(fig)
        
        st.subheader("Pair Plot")
        fig = sns.pairplot(dataset, hue='Species', 
                          palette=color_map)
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(dataset_encoded.corr(), annot=True, cmap='magma', ax=ax)
        st.pyplot(fig)

elif page == "Model Training":
    st.title("Model Training")
    
    if st.button("Train Logistic Regression Model"):
        with st.spinner("Training in progress..."):
            # Preprocess data
            X = dataset_encoded.drop('Species', axis=1)
            y = dataset_encoded['Species']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(max_iter=200)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            
            st.success("Model trained successfully!")
            
            st.header("Model Evaluation")
            
            st.subheader("Accuracy")
            st.write(f"{accuracy_score(y_test, y_pred):.2%}")
            
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), 
                       annot=True, fmt='d', cmap='magma',
                       xticklabels=le.classes_, 
                       yticklabels=le.classes_, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            st.subheader("Cross-Validation Results")
            scoring = {
                'neg_mse': 'neg_mean_squared_error',
                'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))),
                'r2': 'r2'
            }
            
            cv_results = cross_validate(model, scaler.transform(X), y, cv=10, 
                                      scoring=scoring,
                                      return_train_score=True)
            
            results = pd.DataFrame({
                'Fold': range(1, 11),
                'Train_R2': cv_results['train_r2'],
                'Test_R2': cv_results['test_r2'],
                'Train_RMSE': np.sqrt(-cv_results['train_neg_mse']),
                'Test_RMSE': np.sqrt(-cv_results['test_neg_mse'])
            })
            
            st.dataframe(results)
            
            st.write("\nAverage Metrics:")
            st.write(f"Mean Training R2: {results['Train_R2'].mean():.4f}")
            st.write(f"Mean Validation R2: {results['Test_R2'].mean():.4f}")
            
            # Save artifacts
            joblib.dump(model, 'logistic_regression_model.pkl')
            joblib.dump(le, 'iris_encoder.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            st.success("Model artifacts saved successfully!")

elif page == "Prediction":
    st.title("Iris Species Prediction")
    
    try:
        # Load artifacts
        model = joblib.load('logistic_regression_model.pkl')
        le = joblib.load('iris_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        
        st.header("Enter Flower Measurements")
        
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
            sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
        with col2:
            petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
            petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)
        
        if st.button("Predict Species"):
            # Prepare input
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)
            proba = model.predict_proba(input_scaled)
            
            # Display results
            species = le.inverse_transform(prediction)[0]
            st.success(f"Predicted Species: **{species}**")
            
            st.subheader("Prediction Probabilities")
            proba_df = pd.DataFrame({
                'Species': le.classes_,
                'Probability': proba[0]
            }).sort_values('Probability', ascending=False)
            st.dataframe(proba_df)
            
            # Visualize probabilities
            fig, ax = plt.subplots()
            sns.barplot(x='Probability', y='Species', data=proba_df, palette='Blues_d', ax=ax)
            ax.set_title('Prediction Confidence')
            st.pyplot(fig)
            
    except FileNotFoundError:
        st.error("Model not found. Please train the model first from the 'Model Training' page.")