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

def load_data(filepath='Iris.csv'):
    dataset = pd.read_csv(filepath)
    dataset.drop('Id', axis=1, inplace=True)
    dataset.drop_duplicates(inplace=True)
    return dataset

def eda_visualizations(dataset):
    dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].hist()
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.show()
    
    color_map = {'Iris-setosa': 'red', 'Iris-versicolor': 'black', 'Iris-virginica': 'blue'}
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for species, color in color_map.items():
        subset = dataset[dataset['Species'] == species]
        plt.scatter(subset['SepalLengthCm'], subset['SepalWidthCm'], 
                   color=color, label=species)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for species, color in color_map.items():
        subset = dataset[dataset['Species'] == species]
        plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'], 
                   color=color, label=species)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.show()
    
    sns.pairplot(dataset, hue='Species', palette=color_map)
    plt.show()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(dataset.corr(), annot=True, cmap='magma')
    plt.title('Correlation Matrix')
    plt.show()

def preprocess_data(dataset):
    le = LabelEncoder()
    dataset['Species'] = le.fit_transform(dataset['Species'])
    
    X = dataset.drop('Species', axis=1)
    y = dataset['Species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
               annot=True, fmt='d', cmap='magma',
               xticklabels=le.classes_, 
               yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def cross_validation_analysis(model, X, y):
    scoring = {
        'neg_mse': 'neg_mean_squared_error',
        'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))),
        'r2': 'r2'
    }
    
    cv_results = cross_validate(model, X, y, cv=10, 
                              scoring=scoring,
                              return_train_score=True)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Fold': range(1, 11),
        'Train_R2': cv_results['train_r2'],
        'Test_R2': cv_results['test_r2'],
        'Train_RMSE': np.sqrt(-cv_results['train_neg_mse']),
        'Test_RMSE': np.sqrt(-cv_results['test_neg_mse'])
    })
    
    print("\nCross-Validation Results:")
    print(results)
    print("\nAverage Metrics:")
    print(f"Mean Training R2: {results['Train_R2'].mean():.4f}")
    print(f"Mean Validation R2: {results['Test_R2'].mean():.4f}")
    print(f"Mean Training RMSE: {results['Train_RMSE'].mean():.4f}")
    print(f"Mean Validation RMSE: {results['Test_RMSE'].mean():.4f}")

def save_artifacts(model, le, scaler):
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(le, 'iris_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model artifacts saved successfully")

def main():
    # Load data
    dataset = load_data()
    
    # EDA
    eda_visualizations(dataset)
    
    # Preprocess
    X_train, X_test, y_train, y_test, le, scaler = preprocess_data(dataset)
    
    # Train model
    model = train_logistic_regression(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, le)
    
    # Cross-validation
    X_scaled = scaler.transform(dataset.drop('Species', axis=1))
    cross_validation_analysis(model, X_scaled, dataset['Species'])
    
    # Save artifacts
    save_artifacts(model, le, scaler)

if __name__ == "__main__":
    main()