from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import io

from Mall_Dataset import (
    preprocess_data, scale_data, apply_pca, remove_outliers,
    find_optimal_clusters, perform_clustering, CLUSTER_LABELS
)

app = FastAPI()

models = {
    "scaler": None,
    "pca": None,
    "kmeans": None,
    "labels": CLUSTER_LABELS
}

def load_models():
    models["scaler"] = joblib.load("scaler.pkl")
    models["pca"] = joblib.load("pca_model.pkl")
    models["kmeans"] = joblib.load("kmeans_model.pkl")
    try:
        models["labels"] = joblib.load("cluster_labels.pkl")
    except Exception:
        models["labels"] = CLUSTER_LABELS

try:
    load_models()
except Exception:
    pass

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    df = pd.read_csv(io.StringIO((await file.read()).decode("utf-8")))
    processed = preprocess_data(df)
    scaled, scaler = scale_data(processed)
    reduced, pca = apply_pca(scaled)
    reduced = remove_outliers(reduced)
    optimal_k = find_optimal_clusters(reduced)
    kmeans = perform_clustering(reduced, n_clusters=optimal_k)
    clusters = kmeans.labels_
    reduced["Cluster"] = clusters
    cluster_labels = models["labels"]
    reduced["Cluster_Label"] = reduced["Cluster"].map(cluster_labels)
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca_model.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(cluster_labels, "cluster_labels.pkl")
    models["scaler"] = scaler
    models["pca"] = pca
    models["kmeans"] = kmeans
    models["labels"] = cluster_labels
    return JSONResponse({
        "message": "Analysis complete.",
        "optimal_clusters": int(optimal_k),
        "cluster_counts": reduced["Cluster_Label"].value_counts().to_dict(),
        "cluster_labels": cluster_labels
    })

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    scaler = models["scaler"]
    pca = models["pca"]
    kmeans = models["kmeans"]
    labels = models["labels"]
    if not (scaler and pca and kmeans):
        return JSONResponse({"error": "Models not loaded. Run analysis first."}, status_code=500)
    try:
        featnames = ["Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
        required = [float(data[f]) for f in featnames]
        scaled = scaler.transform([required])
        reduced = pca.transform(scaled)
        cluster = int(kmeans.predict(reduced)[0])
        label = labels.get(cluster, f"Cluster {cluster}")
        return {"cluster": cluster, "label": label}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
