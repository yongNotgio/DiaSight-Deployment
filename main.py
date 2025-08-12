import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os

# Load model at startup
MODEL_PATH = os.path.join("models", "stackedblend_predictor.pkl")  # adjust as needed
try:
    predictor = joblib.load(MODEL_PATH)
except Exception as e:
    predictor = None
    print(f"Failed to load model: {e}")

app = FastAPI(title="DiaSight DR Predictor API")

# List all required features here. Adjust as needed for your model.
class PatientInput(BaseModel):
    age: float
    sex: int
    sbp: float
    dbp: float
    hbp: int
    duration: float
    hb1ac: float
    ldl: float
    hdl: float
    chol: float
    urea: float
    bun: float
    uric: float
    egfr: float
    trig: float
    ucr: float
    alt: float
    ast: float

@app.get("/")
def root():
    return {"message": "DiaSight DR Predictor API is running."}

@app.post("/predict")
def predict(input: PatientInput):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        df = pd.DataFrame([input.dict()])
        result = predictor.predict_with_confidence(df)
        return {
            "prediction": int(result["predictions"][0]),
            "class_name": result["class_names"][int(result["predictions"][0])],
            "probabilities": result["probabilities"][0].tolist(),
            "confidence": float(result["confidence"][0]),
            "risk_score": float(result["risk_scores"][0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
