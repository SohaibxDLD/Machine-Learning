import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath='boston.csv'):
    dataset = pd.read_csv(filepath)
    print("\nDataset loaded successfully!")
    print(f"\nShape: {dataset.shape}")
    print("\nFirst 5 rows:")
    print(dataset.head())
    print("\nDataset info:")
    print(dataset.info())
    print("\nMissing values:")
    print(dataset.isnull().sum())
    return dataset

def remove_outliers(dataset):
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset[(dataset >= (Q1 - 1.5 * IQR)) & (dataset <= (Q3 + 1.5 * IQR))]
    return dataset.dropna()

def plot_histograms(dataset, figsize=(20, 15)):
    dataset.hist(figsize=figsize)
    plt.suptitle("Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_boxplots(dataset, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    sns.boxplot(data=dataset)
    plt.title('Box Plot of Features')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.show()

def plot_scatter(dataset, target='MEDV', figsize=(50, 50)):
    features = dataset.columns.drop(target)
    plt.figure(figsize=figsize)
    for i, feature in enumerate(features, 1):
        plt.subplot(4, 4, i)
        plt.scatter(dataset[feature], dataset[target])
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title(f'{feature} vs {target}')
    plt.tight_layout()
    plt.show()

def plot_correlation(dataset, figsize=(12, 10)):
    correlation_matrix = dataset.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='magma', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def prepare_data(dataset, target='MEDV', test_size=0.2, random_state=42):
    X = dataset.drop(target, axis=1)
    y = dataset[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R2 Score: {r2*100:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()
    
    return y_pred

def save_artifacts(model, scaler, model_path='linear_regression_model.pkl', scaler_path='scaler.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def main():
    dataset = load_data()
    dataset = remove_outliers(dataset)
    
    plot_histograms(dataset)
    plot_boxplots(dataset)
    plot_scatter(dataset)
    plot_correlation(dataset)
    
    X_train, X_test, y_train, y_test = prepare_data(dataset)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    model = train_model(X_train_scaled, y_train)
    y_pred = evaluate_model(model, X_test_scaled, y_test)
    
    save_artifacts(model, scaler)

if __name__ == "__main__":
    main()