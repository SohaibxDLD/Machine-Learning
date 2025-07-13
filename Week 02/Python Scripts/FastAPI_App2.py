from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Wine Classification API",
    description="API for predicting wine types using ML models",
)

MODELS = {
    "decision_tree": {
        "model": "model.joblib",
        "scaler": "scaler.joblib"
    },
    "knn": {
        "model": "model2.joblib",
        "scaler": "scaler2.joblib"
    },
    "naive_bayes": {
        "model": "model3.joblib",
        "scaler": "scaler3.joblib"
    }
}

class WineData(BaseModel):
    Alcohol: float
    Malic_acid: float
    Alcalinity: float
    Phenols: float
    Flavanoids: float
    Nonflavanoid_phenols: float
    Proanthocyanins: float
    Hue: float
    OD280_OD315: float
    Proline: float
    model_type: str = "decision_tree"

@app.post("/predict")
async def predict(wine: WineData):
    try:
        model_path = MODELS[wine.model_type]["model"]
        scaler_path = MODELS[wine.model_type]["scaler"]
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        features = np.array([
            wine.Alcohol,
            wine.Malic_acid,
            wine.Alcalinity,
            wine.Phenols,
            wine.Flavanoids,
            wine.Nonflavanoid_phenols,
            wine.Proanthocyanins,
            wine.Hue,
            wine.OD280_OD315,
            wine.Proline
        ]).reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled) if hasattr(model, "predict_proba") else None
        
        return {
            "wine_class": int(prediction[0]),
            "probability": probability.tolist() if probability is not None else None,
            "model_type": wine.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def get_available_models():
    return {"available_models": list(MODELS.keys())}

@app.get("/")
async def health_check():
    return {"status": "healthy"}