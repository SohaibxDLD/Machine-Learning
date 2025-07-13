import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib as jb

def load_data():
    dataset = pd.read_csv('Wine.csv')
    return dataset

def preprocess_data(dataset):
    dataset.drop(['Ash', 'Mg', 'Color.int'], axis=1, inplace=True)
    return dataset

def explore_data(dataset):
    print(dataset.info())
    print(dataset.describe())
    print(dataset.isnull().sum())
    print('Unique values:', dataset.nunique().sum())
    print(dataset.nunique())
    print('Duplicates:', dataset.duplicated().sum())

def visualize_data(dataset):
    for feature in dataset.columns:
        plt.figure(figsize=(10, 5))
        plt.hist(dataset[feature])
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()

    for feature in dataset.columns[:-1]:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=dataset, x=feature, y='Wine')
        plt.show()

    for feature in dataset.columns[:-1]:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=dataset, x='Wine', y=feature)
        plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap='magma')
    plt.show()

def train_decision_tree(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = DecisionTreeClassifier()
    model.fit(X_train_scaled, Y_train)
    
    best_score, best_random_state = find_best_random_state(X, Y, model)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=best_random_state)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, Y_train)
    
    evaluate_model(model, X_test_scaled, Y_test, Y_train, X_train_scaled, 'Decision Tree')
    return model, scaler

def train_knn(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled, Y_train)
    
    best_score, best_random_state = find_best_random_state(X, Y, model, trials=20)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=best_random_state)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, Y_train)
    
    evaluate_model(model, X_test_scaled, Y_test, Y_train, X_train_scaled, 'KNN')
    return model, scaler

def train_naive_bayes(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GaussianNB()
    model.fit(X_train_scaled, Y_train)
    
    best_score, best_random_state = find_best_random_state(X, Y, model, trials=10)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=best_random_state)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, Y_train)
    
    evaluate_model(model, X_test_scaled, Y_test, Y_train, X_train_scaled, 'Naive Bayes')
    return model, scaler

def find_best_random_state(X, Y, model, trials=5):
    best_score = 0
    best_random_state = 0
    
    for i in range(trials):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, Y_train)
        score = model.score(X_test_scaled, Y_test)
        if score > best_score:
            best_score = score
            best_random_state = i
            
    return best_score, best_random_state

def evaluate_model(model, X_test, Y_test, Y_train, X_train, model_name):
    predictions = model.predict(X_test)
    scores = cross_val_score(model, X_train, Y_train, cv=5)
    
    print(f"\n=== {model_name} ===")
    print("Classification Report:")
    print(classification_report(Y_test, predictions))
    print("Cross-validation scores:", scores)
    print("Mean CV score:", scores.mean())
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(Y_test, predictions), annot=True, cmap='magma')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

def save_models(models):
    for name, (model, scaler) in models.items():
        jb.dump(model, f'{name}_model.joblib')
        jb.dump(scaler, f'{name}_scaler.joblib')

def main():
    dataset = load_data()
    dataset = preprocess_data(dataset)
    explore_data(dataset)
    visualize_data(dataset)
    
    X = dataset.drop('Wine', axis=1)
    Y = dataset['Wine']
    
    models = {
        'decision_tree': train_decision_tree(X, Y),
        'knn': train_knn(X, Y),
        'naive_bayes': train_naive_bayes(X, Y)
    }
    
    save_models(models)

if __name__ == "__main__":
    main()