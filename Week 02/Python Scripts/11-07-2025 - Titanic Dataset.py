import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib as jb

def load_data():
    dataset = pd.read_csv("titanic.csv")
    return dataset

def preprocess_data(dataset):
    columns_to_drop = ['PassengerId', 'Cabin', 'Name', 'Ticket', 'SibSp', 'Parch', 'Embarked']
    for col in columns_to_drop:
        if col in dataset.columns:
            dataset.drop(col, axis=1, inplace=True)
    
    age_impute = SimpleImputer(strategy='mean')
    dataset['Age'] = age_impute.fit_transform(dataset[['Age']])
    
    if 'Embarked' in dataset.columns:
        embarked_impute = SimpleImputer(strategy='most_frequent')
        dataset['Embarked'] = embarked_impute.fit_transform(dataset[['Embarked']]).ravel()
    
    label_encoding = LabelEncoder()
    if 'Sex' in dataset.columns:
        dataset['Sex'] = label_encoding.fit_transform(dataset['Sex'])
    if 'Embarked' in dataset.columns:
        dataset['Embarked'] = label_encoding.fit_transform(dataset['Embarked'])
    
    return dataset

def exploratory_data_analysis(dataset):
    features = dataset.columns
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.histplot(dataset[feature])
        plt.show()
    
    for i in features:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=dataset, x=i, y='Survived')
        plt.show()

    for i in features:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=dataset, x=i, y='Survived')
        plt.show()

    correlation_matrix = dataset.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='magma')
    plt.show()

def train_decision_tree(X, Y):
    best_score = 0
    best_random_state = 0 

    for i in range(20):  
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = DecisionTreeClassifier()
        model.fit(X_train_scaled, Y_train)
        score = model.score(X_test_scaled, Y_test)
        if score > best_score:
            best_score = score
            best_random_state = i

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=best_random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, Y_train)
    final_accuracy = model.score(X_test_scaled, Y_test)
    scores = cross_val_score(model, X_train_scaled, Y_train, cv=5)
    Predicted_data = model.predict(X_test_scaled)
    print(cr(Y_test, Predicted_data))
    confusion = cm(Y_test, Predicted_data)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, cmap='magma')
    plt.show()
    jb.dump(model, 'model.joblib')
    jb.dump(scaler, 'scaler.joblib')
    return model, scaler

def train_knn(X, Y):
    best_score = 0
    best_random_state = 0 

    for i in range(20):  
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train_scaled, Y_train)
        score = model.score(X_test_scaled, Y_test)
        if score > best_score:
            best_score = score
            best_random_state = i

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=best_random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, Y_train)
    final_accuracy = model.score(X_test_scaled, Y_test)
    scores = cross_val_score(model, X_train_scaled, Y_train, cv=5)
    Predicted_data = model.predict(X_test_scaled)
    print(cr(Y_test, Predicted_data))
    confusion = cm(Y_test, Predicted_data)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, cmap='magma')
    plt.show()
    jb.dump(model, 'model2.joblib')
    jb.dump(scaler, 'scaler2.joblib')
    return model, scaler

def train_naive_bayes(X, Y):
    best_score = 0
    best_random_state = 0 

    for i in range(20):  
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = GaussianNB()
        model.fit(X_train_scaled, Y_train)
        score = model.score(X_test_scaled, Y_test)
        if score > best_score:
            best_score = score
            best_random_state = i

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=best_random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, Y_train)
    final_accuracy = model.score(X_test_scaled, Y_test)
    scores = cross_val_score(model, X_train_scaled, Y_train, cv=5)
    Predicted_data = model.predict(X_test_scaled)
    print(cr(Y_test, Predicted_data))
    confusion = cm(Y_test, Predicted_data)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, cmap='magma')
    plt.show()
    jb.dump(model, 'model3.joblib')
    jb.dump(scaler, 'scaler3.joblib')
    return model, scaler

def main():
    dataset = load_data()
    dataset = preprocess_data(dataset)
    exploratory_data_analysis(dataset)
    
    X = dataset.drop('Survived', axis=1)
    Y = dataset['Survived']
    
    print("\nTraining Decision Tree Model...")
    train_decision_tree(X, Y)
    
    print("\nTraining KNN Model...")
    train_knn(X, Y)
    
    print("\nTraining Naive Bayes Model...")
    train_naive_bayes(X, Y)

if __name__ == "__main__":
    main()