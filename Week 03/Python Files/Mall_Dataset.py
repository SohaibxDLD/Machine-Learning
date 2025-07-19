import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib   

# OPTIONAL: Adjust descriptive labels after reviewing your own data.
CLUSTER_LABELS = {
    0: "High Value Shoppers",
    1: "Careful Spenders",
    2: "Young Low Spenders",
    3: "Target Potential"
}

def load_data(file_path):
    dataset = pd.read_csv(file_path)
    print("\nDataset Info:")
    print(dataset.info())
    print("\nDataset Description:")
    print(dataset.describe())
    return dataset

def preprocess_data(dataset):
    print("\nChecking for duplicates:")
    print(f"Sum of duplicated values: {dataset.duplicated().sum()}")
    print("\nChecking for null values:")
    print(f"Sum of null values:\n{dataset.isnull().sum()}")
    le = LabelEncoder()
    if 'Genre' in dataset.columns:
        dataset['Genre'] = le.fit_transform(dataset['Genre'])
    if 'CustomerID' in dataset.columns:
        dataset = dataset.drop('CustomerID', axis=1)
    return dataset

def scale_data(dataset):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, scaler

def apply_pca(scaled_data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{i+1}' for i in range(n_components)])
    return pca_df, pca

def explore_data(dataset):
    for col in dataset.columns:
        sns.displot(dataset[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataset)
    plt.title('Box Plot of Features')
    plt.show()
    corr_matrix = dataset.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def remove_outliers(dataset):
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    filtered = dataset[~((dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f"\nOriginal shape: {dataset.shape}, New shape after outlier removal: {filtered.shape}")
    return filtered

def find_optimal_clusters(data, max_k=10):
    inertia = []
    silhouette_scores = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        if k >= 2:
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_k+1), inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k+1), silhouette_scores, 'go-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.tight_layout()
    plt.show()
    optimal_k = np.argmax(silhouette_scores) + 2
    print(f"Suggested optimal number of clusters: {optimal_k}")
    return optimal_k

def perform_clustering(data, n_clusters, init='k-means++', max_iter=300):
    kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=42)
    kmeans.fit(data)
    return kmeans

def evaluate_classifier(X, y, classifier, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = classifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return model

def main():
    data = load_data('Mall_Customers.csv')
    data = preprocess_data(data)
    scaled_data, scaler = scale_data(data)
    pca_data, pca = apply_pca(scaled_data)
    explore_data(pca_data)
    pca_data = remove_outliers(pca_data)
    optimal_k = find_optimal_clusters(pca_data)
    kmeans = perform_clustering(pca_data, n_clusters=optimal_k)
    pca_data['Cluster'] = kmeans.labels_

    # Add labels
    pca_data['Cluster_Label'] = pca_data['Cluster'].map(CLUSTER_LABELS)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Label', data=pca_data, palette='viridis')
    plt.title('Customer Clusters (Labeled)')
    plt.show()

    original_data = data.iloc[pca_data.index].copy()
    original_data['Cluster'] = kmeans.labels_
    original_data['Cluster_Label'] = original_data['Cluster'].map(CLUSTER_LABELS)
    X = original_data.drop(['Cluster', 'Cluster_Label'], axis=1)
    y = original_data['Cluster']

    print("\nEvaluating Decision Tree:")
    dt_model = evaluate_classifier(X, y, DecisionTreeClassifier)
    print("\nEvaluating Logistic Regression:")
    lr_model = evaluate_classifier(X, y, lambda: LogisticRegression(max_iter=1000))
    print("\nEvaluating Random Forest:")
    rf_model = evaluate_classifier(X, y, RandomForestClassifier)

    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca_model.pkl')
    joblib.dump(dt_model, 'decision_tree.pkl')
    joblib.dump(lr_model, 'logistic_regression.pkl')
    joblib.dump(rf_model, 'random_forest.pkl')
    # Optionally save labels
    joblib.dump(CLUSTER_LABELS, 'cluster_labels.pkl')   

if __name__ == "__main__":
    main()
