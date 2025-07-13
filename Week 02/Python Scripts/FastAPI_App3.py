from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Breast Cancer Diagnosis API",
    description="API for predicting breast cancer diagnosis using ML models"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {
    "decision_tree": {
        "model": "model.joblib",
        "scaler": "scaler.joblib"
    },
    "knn": {
        "model": "model.joblib",
        "scaler": "scaler.joblib"
    },
    "naive_bayes": {
        "model": "model.joblib",
        "scaler": "scaler.joblib"
    }
}

class CancerData(BaseModel):
    # These are the EXACT 19 features your model expects
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    # Note: fractal_dimension_se is intentionally omitted to match your 19-feature model
    model_type: str = "decision_tree"

@app.post("/predict")
async def predict(cancer: CancerData):
    try:
        model = joblib.load(MODELS[cancer.model_type]["model"])
        scaler = joblib.load(MODELS[cancer.model_type]["scaler"])
        
        # Create array with EXACTLY 19 features in the correct order
        features = np.array([
            cancer.radius_mean,
            cancer.texture_mean,
            cancer.perimeter_mean,
            cancer.area_mean,
            cancer.smoothness_mean,
            cancer.compactness_mean,
            cancer.concavity_mean,
            cancer.concave_points_mean,
            cancer.symmetry_mean,
            cancer.fractal_dimension_mean,
            cancer.radius_se,
            cancer.texture_se,
            cancer.perimeter_se,
            cancer.area_se,
            cancer.smoothness_se,
            cancer.compactness_se,
            cancer.concavity_se,
            cancer.concave_points_se,
            cancer.symmetry_se
            # Omit fractal_dimension_se to keep 19 features
        ]).reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled) if hasattr(model, "predict_proba") else None
        
        return {
            "diagnosis": "Malignant" if prediction[0] == 1 else "Benign",
            "probability": probability.tolist() if probability is not None else None,
            "model_type": cancer.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def get_models():
    return {"available_models": list(MODELS.keys())}

@app.get("/")
async def health_check():
    return {"status": "API is healthy"}