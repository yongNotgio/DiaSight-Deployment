# main.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pickle

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# -------------------------------
# 1) Predictor class (identical to original)
# -------------------------------
class DiabetesRiskPredictor:
    """
    Unified model wrapper for deployment with Streamlit/FastAPI
    Handles all preprocessing, feature engineering, and prediction steps
    """
    def __init__(self):
        self.model = None
        self.model_type = None
        self.tau0 = None
        self.tau1 = None
        self.temperature = 1.0
        self.selected_features = None
        self.feature_stats = None  # For validation
        self.scaler = None  # For models that need scaling
        self.label_encoders = None  # For categorical features
        self.is_ensemble = False

        # For ensemble models
        self.catboost_model = None
        self.elasticnet_model = None
        self.weight_catboost = None
        self.weight_elasticnet = None
        self.temperature_catboost = 1.0
        self.temperature_elasticnet = 1.0

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'age' in df.columns and 'hb1ac' in df.columns:
            df['age_hba1c_interaction'] = df['age'] * df['hb1ac']
        elif 'age' in df.columns and 'hba1c' in df.columns:
            df['age_hba1c_interaction'] = df['age'] * df['hba1c']
        if 'duration' in df.columns and 'egfr' in df.columns:
            df['duration_egfr_ratio'] = df['duration'] / (df['egfr'] + 1e-6)
        if 'ldl' in df.columns and 'hdl' in df.columns:
            df['ldl_hdl_ratio'] = df['ldl'] / (df['hdl'] + 1e-6)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _apply_temperature_scaling(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        if temperature == 1.0:
            return probs
        try:
            logits = np.log(probs + 1e-15)
            calibrated_logits = logits / temperature
            exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits, axis=1, keepdims=True))
            calibrated_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return calibrated_probs
        except:
            return probs

    def _apply_thresholds(self, probs: np.ndarray) -> np.ndarray:
        p0, p1, p2 = probs.T
        preds = np.where(p0 > self.tau0, 0, np.where(p1 > self.tau1, 1, 2))
        return preds

    def _preprocess_input(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = X.copy()
        if self.scaler is not None:
            try:
                if hasattr(self.scaler, 'feature_names_in_'):
                    scale_features = [f for f in self.scaler.feature_names_in_ if f in X_processed.columns]
                else:
                    scale_features = self.selected_features if self.selected_features else X_processed.columns
                if scale_features:
                    X_processed[scale_features] = self.scaler.transform(X_processed[scale_features])
            except Exception as e:
                print(f"Warning: Scaling failed, using unscaled data: {e}")
        if self.label_encoders is not None:
            for feature, encoder in self.label_encoders.items():
                if feature in X_processed.columns:
                    try:
                        X_processed[feature] = encoder.transform(X_processed[feature])
                    except Exception as e:
                        print(f"Warning: Label encoding failed for {feature}: {e}")
        return X_processed

    def predict_proba(self, X: Union[pd.DataFrame, dict]) -> np.ndarray:
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = X.to_frame().T
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame, Series, or dictionary")
        X_processed = self._engineer_features(X)

        if self.selected_features:
            missing_features = set(self.selected_features) - set(X_processed.columns)
            if missing_features:
                for feat in missing_features:
                    if self.feature_stats and feat in self.feature_stats.get('means', {}):
                        X_processed[feat] = self.feature_stats['means'][feat]
                    else:
                        X_processed[feat] = 0
            X_processed = X_processed[self.selected_features]

        if self.feature_stats:
            for col in X_processed.columns:
                if col in self.feature_stats.get('means', {}):
                    X_processed[col] = X_processed[col].fillna(self.feature_stats['means'][col])
        X_processed = X_processed.fillna(0)
        X_processed = self._preprocess_input(X_processed)

        if self.is_ensemble:
            catboost_probs = self.catboost_model.predict_proba(X_processed)
            elasticnet_probs = self.elasticnet_model.predict_proba(X_processed)
            catboost_probs = self._apply_temperature_scaling(catboost_probs, self.temperature_catboost)
            elasticnet_probs = self._apply_temperature_scaling(elasticnet_probs, self.temperature_elasticnet)
            probs = self.weight_catboost * catboost_probs + self.weight_elasticnet * elasticnet_probs
        else:
            probs = self.model.predict_proba(X_processed)
            probs = self._apply_temperature_scaling(probs, self.temperature)
        return probs

    def predict(self, X: Union[pd.DataFrame, dict]) -> np.ndarray:
        probs = self.predict_proba(X)
        return self._apply_thresholds(probs)

    def predict_with_confidence(self, X: Union[pd.DataFrame, dict]) -> Dict:
        probs = self.predict_proba(X)
        preds = self._apply_thresholds(probs)
        results = {
            'predictions': preds,
            'probabilities': probs,
            'confidence': np.max(probs, axis=1),
            'risk_scores': probs[:, 2] * 100,  # Severe DR risk as percentage
            'class_names': ['No DR', 'Mild DR', 'Severe DR']
        }
        return results

    def get_feature_importance(self) -> Dict:
        if self.model_type == "CatBoost" and hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            return dict(zip(self.selected_features, importance))
        elif hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.selected_features, self.model.feature_importances_))
        else:
            return {}

# -------------------------------
# 2) Pydantic Models for API
# -------------------------------
class PatientInput(BaseModel):
    """Input model for single patient prediction"""
    age: float = Field(..., description="Patient age")
    hb1ac: float = Field(..., description="HbA1c level")
    duration: float = Field(..., description="Duration of diabetes")
    egfr: float = Field(..., description="Estimated glomerular filtration rate")
    ldl: float = Field(..., description="LDL cholesterol")
    hdl: float = Field(..., description="HDL cholesterol")
    chol: float = Field(..., description="Total cholesterol")
    sbp: float = Field(..., description="Systolic blood pressure")
    dbp: float = Field(..., description="Diastolic blood pressure")
    hbp: int = Field(..., description="Hypertension (0/1)")
    sex: int = Field(..., description="Sex (0/1/2)")
    uric: float = Field(..., description="Uric acid")
    bun: float = Field(..., description="Blood urea nitrogen")
    urea: float = Field(..., description="Urea")
    trig: float = Field(..., description="Triglycerides")
    ucr: float = Field(..., description="Urine creatinine")
    alt: float = Field(..., description="ALT")
    ast: float = Field(..., description="AST")

    class Config:
        extra = "allow"  # Allow additional fields

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    prediction_index: int
    confidence: float
    risk_score: float
    probabilities: Dict[str, float]
    class_names: List[str]

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_processed: int

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    selected_features: List[str]
    class_names: List[str]
    is_ensemble: bool
    model_type: Optional[str]
    tau0: Optional[float]
    tau1: Optional[float]

class FeatureImportance(BaseModel):
    """Feature importance response"""
    feature_importance: Dict[str, float]
    available: bool

# -------------------------------
# 3) FastAPI Application
# -------------------------------
app = FastAPI(
    title="DiaSight - Diabetes Retinopathy Risk Prediction API",
    description="API for predicting diabetes retinopathy risk using machine learning models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 4) Global variables and utilities
# -------------------------------
# Default models directory - you can change this
DEFAULT_MODELS_DIR = r"C:\Users\gioan\Documents\GitHub\DiaSight-Deployment\models"
models_dir = Path(DEFAULT_MODELS_DIR)
loaded_predictor: Optional[DiabetesRiskPredictor] = None
current_model_name: Optional[str] = None
meta_json: Optional[Dict] = None
stats_json: Optional[Dict] = None

def load_metadata():
    """Load metadata and statistics if available"""
    global meta_json, stats_json
    
    meta_path = models_dir / "model_metadata.json"
    stats_path = models_dir / "feature_statistics.json"
    
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta_json = json.load(f)
        except Exception:
            meta_json = None
    
    if stats_path.exists():
        try:
            with open(stats_path, "r") as f:
                stats_json = json.load(f)
        except Exception:
            stats_json = None

def list_available_models() -> List[str]:
    """List available predictor models"""
    if not models_dir.exists():
        return []
    return sorted([f.name for f in models_dir.glob("*_predictor.pkl")])

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Load metadata on startup
load_metadata()

# -------------------------------
# 5) API Endpoints
# -------------------------------

@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "DiaSight - Diabetes Retinopathy Risk Prediction API",
        "version": "1.0.0",
        "status": "running",
        "current_model": current_model_name,
        "available_models": list_available_models()
    }

@app.get("/models", summary="List available models")
async def get_available_models():
    """Get list of available model files"""
    models = list_available_models()
    return {
        "available_models": models,
        "total_count": len(models),
        "current_model": current_model_name
    }

@app.post("/load-model/{model_name}", summary="Load a specific model")
async def load_model(model_name: str):
    """Load a specific model by name"""
    global loaded_predictor, current_model_name
    
    if not model_name.endswith("_predictor.pkl"):
        model_name += "_predictor.pkl"
    
    model_path = models_dir / model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        loaded_predictor = joblib.load(model_path)
        current_model_name = model_name
        return {
            "message": f"Successfully loaded model: {model_name}",
            "model_name": model_name,
            "is_ensemble": getattr(loaded_predictor, 'is_ensemble', False),
            "model_type": getattr(loaded_predictor, 'model_type', None)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/model-info", response_model=ModelInfo, summary="Get current model information")
async def get_model_info():
    """Get information about the currently loaded model"""
    if loaded_predictor is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")
    
    # Get selected features
    selected_features = getattr(loaded_predictor, 'selected_features', None)
    if not selected_features and meta_json and 'feature_names' in meta_json:
        selected_features = meta_json['feature_names']
    if not selected_features:
        selected_features = ["age","hb1ac","duration","egfr","ldl","hdl","chol","sbp","dbp","hbp","sex","uric","bun","urea","trig","ucr","alt","ast"]
    
    # Get class names
    class_names = ['No DR', 'Mild DR', 'Severe DR']
    if meta_json and 'class_names' in meta_json:
        class_names = meta_json['class_names']
    
    return ModelInfo(
        model_name=current_model_name or "unknown",
        selected_features=selected_features,
        class_names=class_names,
        is_ensemble=getattr(loaded_predictor, 'is_ensemble', False),
        model_type=getattr(loaded_predictor, 'model_type', None),
        tau0=getattr(loaded_predictor, 'tau0', None),
        tau1=getattr(loaded_predictor, 'tau1', None)
    )

@app.get("/feature-importance", response_model=FeatureImportance, summary="Get feature importance")
async def get_feature_importance():
    """Get feature importance from the current model"""
    if loaded_predictor is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")
    
    try:
        importance = loaded_predictor.get_feature_importance()
        return FeatureImportance(
            feature_importance=convert_numpy_types(importance),
            available=len(importance) > 0
        )
    except Exception as e:
        return FeatureImportance(
            feature_importance={},
            available=False
        )

@app.post("/predict", response_model=PredictionResponse, summary="Make a single prediction")
async def predict_single(
    patient_input: PatientInput,
    tau0: Optional[float] = Query(None, description="Override tau0 threshold"),
    tau1: Optional[float] = Query(None, description="Override tau1 threshold")
):
    """Make a prediction for a single patient"""
    if loaded_predictor is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")
    
    # Override thresholds if provided
    if tau0 is not None:
        loaded_predictor.tau0 = tau0
    if tau1 is not None:
        loaded_predictor.tau1 = tau1
    
    try:
        # Convert pydantic model to dict
        input_dict = patient_input.dict()
        
        # Make prediction
        result = loaded_predictor.predict_with_confidence(input_dict)
        
        # Convert numpy types for JSON serialization
        result = convert_numpy_types(result)
        
        pred_idx = int(result['predictions'][0])
        probs = result['probabilities'][0]
        class_names = result.get('class_names', ['No DR', 'Mild DR', 'Severe DR'])
        
        # Create probability dictionary
        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        
        return PredictionResponse(
            prediction=class_names[pred_idx],
            prediction_index=pred_idx,
            confidence=float(result['confidence'][0]),
            risk_score=float(result['risk_scores'][0]),
            probabilities=prob_dict,
            class_names=class_names
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse, summary="Make batch predictions")
async def predict_batch(
    file: UploadFile = File(..., description="CSV file with patient data"),
    tau0: Optional[float] = Query(None, description="Override tau0 threshold"),
    tau1: Optional[float] = Query(None, description="Override tau1 threshold")
):
    """Make predictions for multiple patients from a CSV file"""
    if loaded_predictor is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Override thresholds if provided
    if tau0 is not None:
        loaded_predictor.tau0 = tau0
    if tau1 is not None:
        loaded_predictor.tau1 = tau1
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        predictions = []
        class_names = ['No DR', 'Mild DR', 'Severe DR']
        
        for _, row in df.iterrows():
            try:
                result = loaded_predictor.predict_with_confidence(row.to_dict())
                result = convert_numpy_types(result)
                
                pred_idx = int(result['predictions'][0])
                probs = result['probabilities'][0]
                current_class_names = result.get('class_names', class_names)
                
                # Create probability dictionary
                prob_dict = {current_class_names[i]: float(probs[i]) for i in range(len(current_class_names))}
                
                predictions.append(PredictionResponse(
                    prediction=current_class_names[pred_idx],
                    prediction_index=pred_idx,
                    confidence=float(result['confidence'][0]),
                    risk_score=float(result['risk_scores'][0]),
                    probabilities=prob_dict,
                    class_names=current_class_names
                ))
            except Exception as e:
                # Handle individual row errors gracefully
                predictions.append(PredictionResponse(
                    prediction="Error",
                    prediction_index=-1,
                    confidence=0.0,
                    risk_score=0.0,
                    probabilities={"Error": 1.0},
                    class_names=["Error"]
                ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/health", summary="Health check")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": loaded_predictor is not None,
        "current_model": current_model_name,
        "models_directory_exists": models_dir.exists()
    }

# -------------------------------
# 6) Startup and Configuration
# -------------------------------

@app.on_event("startup")
async def startup_event():
    """Load default model on startup if available"""
    global loaded_predictor, current_model_name
    
    # Try to load the stacked blend model by default
    default_model = "stackedblend_predictor.pkl"
    available_models = list_available_models()
    
    if default_model in available_models:
        try:
            loaded_predictor = joblib.load(models_dir / default_model)
            current_model_name = default_model
            print(f"✅ Loaded default model: {default_model}")
        except Exception as e:
            print(f"❌ Failed to load default model: {e}")
    elif available_models:
        # Load the first available model
        try:
            first_model = available_models[0]
            loaded_predictor = joblib.load(models_dir / first_model)
            current_model_name = first_model
            print(f"✅ Loaded first available model: {first_model}")
        except Exception as e:
            print(f"❌ Failed to load first model: {e}")
    else:
        print("⚠️ No models found in directory")

# -------------------------------
# 7) Main execution
# -------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )