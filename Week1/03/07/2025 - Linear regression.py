import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_and_inspect_data(filepath='boston.csv'):
    print("\nLOADING THE DATA ⏳")
    dataset = pd.read_csv(filepath)
    print("\nFirst 5 rows:")
    print(dataset.head())
    print("\nDataset info:")
    print(dataset.info())
    print("\nMissing values:")
    print(dataset.isnull().sum())
    return dataset

def perform_eda(dataset):
    print("\nEXPLORATORY DATA ANALYSIS (EDA) 📈📊")
    
    print("\nDescriptive statistics:")
    print(dataset.describe())
    
    print("\nPlotting histograms...")
    dataset.hist(figsize=(20,15))
    plt.show()
    
    print("\nPlotting boxplots...")
    plt.figure(figsize=(10,8))
    sns.boxplot(data=dataset)
    plt.title('Box Plot of Features')
    plt.xlabel('features')
    plt.ylabel('Values')
    plt.show()
    
    print("\nPlotting scatter plots...")
    plt.figure(figsize=(12, 12))
    features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
               'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    for i in range(len(features)):
        plt.subplot(4, 4, i + 1)
        plt.scatter(dataset[features[i]], dataset['MEDV'])
        plt.xlabel(features[i])
        plt.ylabel('MEDV')
        plt.title(f'{features[i]} vs MEDV')
    plt.tight_layout()
    plt.show()
    
    print("\nCalculating correlation matrix...")
    correlation_matrix = dataset.corr()
    print(correlation_matrix)
    
    print("\nPlotting correlation heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='magma')
    plt.title('Correlation Matrix Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.show()

def train_and_evaluate(dataset):
    print("\nTRAINING THE MODEL 🏋💪")
    
    X = dataset.drop('MEDV', axis=1)
    Y = dataset['MEDV']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    
    print("\nOriginal Values (sample):", Y_test.values[:5])
    print("Predicted Values (sample):", Y_pred[:5])
    
    print("\nEVALUATION METRICS 🗿")
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2_score = model.score(X_test, Y_test)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R2 Score: {r2_score*100:.2f}%")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(Y_test, Y_pred, color='red')
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '-', color='blue')
    plt.title('Predicted vs Original Values')
    plt.xlabel('Original Values')
    plt.ylabel('Predicted Values')
    plt.show()

def main():
    data = load_and_inspect_data()
    perform_eda(data)
    train_and_evaluate(data)

if __name__ == "__main__":
    main()