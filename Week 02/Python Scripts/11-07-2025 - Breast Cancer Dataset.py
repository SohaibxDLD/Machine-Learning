import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import joblib as jb

def load_data():
    dataset = pd.read_csv('breast-cancer.csv')
    if 'id' in dataset.columns:
        dataset = dataset.drop('id', axis=1)
    return dataset

def preprocess_data(dataset):
    le = LabelEncoder()
    dataset['diagnosis'] = le.fit_transform(dataset['diagnosis'])
    
    features = dataset.columns
    for feature in features:
        q1 = dataset[feature].quantile(0.25)
        q3 = dataset[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        dataset = dataset[(dataset[feature] > lower_bound) & (dataset[feature] < upper_bound)]
    
    features_drop = ['smoothness_mean', 'symmetry_mean', 'fractal_dimension_mean', 
                    'texture_se', 'smoothness_se', 'compactness_se', 'symmetry_se', 
                    'fractal_dimension_se', 'smoothness_worst', 'symmetry_worst', 
                    'fractal_dimension_worst']
    dataset = dataset.drop(columns=[col for col in features_drop if col in dataset.columns])
    
    return dataset

def explore_data(dataset):
    print(dataset.info())
    print(dataset.describe())
    print("Null values:\n", dataset.isnull().sum())
    print("Unique values:", dataset.nunique().sum())
    print("Duplicates:", dataset.duplicated().sum())

def visualize_data(dataset):
    features = dataset.columns
    
    for feature in features:
        plt.figure(figsize=(10, 5))
        plt.hist(dataset[feature])
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()

    for feature in features[:-1]:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=dataset, x=feature, y='diagnosis')
        plt.xlabel(feature)
        plt.ylabel('Diagnosis')
        plt.show()

    for feature in features[:-1]:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=dataset, x='diagnosis', y=feature)
        plt.xlabel('Diagnosis')
        plt.ylabel(feature)
        plt.show()

    plt.figure(figsize=(20, 15))
    sns.heatmap(dataset.corr(), annot=True, cmap='magma')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def train_model(X, Y, model_type, model_name, n_neighbors=3):
    best_score = 0
    best_random_state = 0
    trials = 20 if model_type == 'knn' else (10 if model_type == 'naive_bayes' else 5)
    
    for i in range(trials):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'decision_tree':
            model = DecisionTreeClassifier()
        elif model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_type == 'naive_bayes':
            model = GaussianNB()
        
        model.fit(X_train_scaled, Y_train)
        score = model.score(X_test_scaled, Y_test)
        
        if score > best_score:
            best_score = score
            best_random_state = i
            best_scaler = scaler
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=best_random_state)
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)
    model.fit(X_train_scaled, Y_train)
    
    evaluate_model(model, X_test_scaled, Y_test, model_name)
    return model, best_scaler

def evaluate_model(model, X_test, Y_test, model_name):
    predictions = model.predict(X_test)
    
    print(f"\n=== {model_name} ===")
    print("Classification Report:")
    print(classification_report(Y_test, predictions))
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(Y_test, predictions), annot=True, cmap='magma', fmt='g')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
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
    
    X = dataset.drop('diagnosis', axis=1)
    Y = dataset['diagnosis']
    
    models = {
        'decision_tree': train_model(X, Y, 'decision_tree', 'Decision Tree'),
        'knn': train_model(X, Y, 'knn', 'KNN'),
        'naive_bayes': train_model(X, Y, 'naive_bayes', 'Naive Bayes')
    }
    
    save_models(models)

if __name__ == "__main__":
    main()