from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting Titanic passenger survival",
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

class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    model_type: str = "decision_tree"

@app.post("/predict")
async def predict(passenger: PassengerData):
    try:
        model_path = MODELS[passenger.model_type]["model"]
        scaler_path = MODELS[passenger.model_type]["scaler"]
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        sex_encoded = 1 if passenger.Sex.lower() == "male" else 0
        features = np.array([
            passenger.Pclass,
            sex_encoded,
            passenger.Age,
            passenger.Fare
        ]).reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled) if hasattr(model, "predict_proba") else None
        
        return {
            "survived": bool(prediction[0]),
            "probability": probability.tolist() if probability is not None else None,
            "model_type": passenger.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def get_available_models():
    return {"available_models": list(MODELS.keys())}

@app.get("/")
async def health_check():
    return {"status": "healthy"}