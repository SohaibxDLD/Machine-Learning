import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset"""
    print("\nLOADING THE DATA ⏳")
    dataset = pd.read_csv(filepath)
    
    print("\nFirst 5 rows:")
    print(dataset.head())
    
    print("\nDataset info:")
    print(dataset.info())
    
    print("\nMissing values:")
    print(dataset.isnull().sum())
    
    dataset.drop('Id', axis=1, inplace=True)
    
    return dataset

def perform_eda(dataset):
    """Perform exploratory data analysis"""
    print("\nEXPLORATORY DATA ANALYSIS (EDA) 📈📊")
    
    print("\nDescriptive statistics:")
    print(dataset.describe())
    
    print("\nPlotting histograms...")
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    dataset['SepalLengthCm'].hist()
    plt.title('Sepal Length')
    
    plt.subplot(2, 2, 2)
    dataset['SepalWidthCm'].hist()
    plt.title('Sepal Width')
    
    plt.subplot(2, 2, 3)
    dataset['PetalLengthCm'].hist()
    plt.title('Petal Length')
    
    plt.subplot(2, 2, 4)
    dataset['PetalWidthCm'].hist()
    plt.title('Petal Width')
    
    plt.tight_layout()
    plt.show()
    
    print("\nPlotting scatter plots...")
    color_of_classes = ['red', 'black', 'blue']
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(3):
        feature = dataset[dataset['Species'] == classes[i]]
        plt.scatter(feature['SepalLengthCm'], feature['SepalWidthCm'], 
                   color=color_of_classes[i], label=classes[i])
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.title('Sepal Length vs Width')
    
    plt.subplot(1, 2, 2)
    for i in range(3):
        feature = dataset[dataset['Species'] == classes[i]]
        plt.scatter(feature['PetalLengthCm'], feature['PetalWidthCm'], 
                   color=color_of_classes[i], label=classes[i])
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.title('Petal Length vs Width')
    
    plt.tight_layout()
    plt.show()
    
    print("\nPlotting pairplot...")
    sns.pairplot(dataset, hue='Species', 
                palette={'Iris-setosa': 'red', 
                         'Iris-versicolor': 'black', 
                         'Iris-virginica': 'blue'})
    plt.show()
    
    print("\nCalculating correlation matrix...")
    correlation_matrix = dataset.drop('Species', axis=1).corr()
    print(correlation_matrix)
    
    print("\nPlotting correlation heatmap...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='magma')
    plt.title('Correlation Matrix Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.show()

def train_and_evaluate_model(dataset):
    """Train and evaluate the logistic regression model"""
    print("\nTRAINING THE MODEL 🏋💪")
    
    label_encoder = LabelEncoder()
    dataset['Species'] = label_encoder.fit_transform(dataset['Species'])
    print("\nDataset after label encoding:")
    print(dataset.head())
    
    X = dataset.drop('Species', axis=1)
    Y = dataset['Species']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    
    print("\nOriginal values of the test set (sample):", Y_test.values[:10])
    print("Predicted values of the model (sample):", Y_pred[:10])
    
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"\nAccuracy of the model: {accuracy * 100:.2f}%")
    
    print("\nCLASSIFICATION REPORT AND CONFUSION MATRIX 🗿")
    print("\nClassification report:")
    print(classification_report(Y_test, Y_pred, target_names=label_encoder.classes_))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    path = 'Iris.csv'
    data = load_and_preprocess_data()
    perform_eda(data)
    train_and_evaluate_model(data)

if __name__ == "__main__":
    main()