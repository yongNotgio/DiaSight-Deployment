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
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import uvicorn

# -------------------------------
# 1) Import Predictor class from separate module
# -------------------------------
from predictor import DiabetesRiskPredictor

# -------------------------------
# Lifespan Events
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    global loaded_predictor, current_model_name
    
    # Try to load models in order of preference
    preferred_models = [
        "stacked_blend_model.pkl",
        "stackedblend_predictor.pkl", 
        "catboost_model.pkl",
        "catboost_predictor.pkl",
        "lightgbm_model.pkl"
    ]
    
    available_models = list_available_models()
    
    for model_name in preferred_models:
        if model_name in available_models:
            try:
                model_path = models_dir / model_name
                if model_name.endswith("_predictor.pkl"):
                    loaded_predictor = joblib.load(model_path)
                else:
                    loaded_predictor = create_predictor_from_model(model_path)
                
                current_model_name = model_name
                print(f"✅ Loaded default model: {model_name}")
                break
            except Exception as e:
                print(f"❌ Failed to load model {model_name}: {e}")
                continue
    else:
        # If no preferred model found, try the first available
        if available_models:
            try:
                first_model = available_models[0]
                model_path = models_dir / first_model
                if first_model.endswith("_predictor.pkl"):
                    loaded_predictor = joblib.load(model_path)
                else:
                    loaded_predictor = create_predictor_from_model(model_path)
                
                current_model_name = first_model
                print(f"✅ Loaded first available model: {first_model}")
            except Exception as e:
                print(f"❌ Failed to load first model: {e}")
        else:
            print("⚠️ No models found in directory")
    
    yield
    # Shutdown - cleanup if needed
    print("🔄 Application shutting down")

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
    version="1.0.0",
    lifespan=lifespan
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
# Default models directory - use relative path for deployment
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
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

def create_predictor_from_model(model_path: Path) -> DiabetesRiskPredictor:
    """Create a DiabetesRiskPredictor from a raw model file"""
    predictor = DiabetesRiskPredictor()
    
    # Load the raw model
    model_data = joblib.load(model_path)
    
    # Load feature statistics and metadata
    if stats_json:
        predictor.feature_stats = stats_json
    
    if meta_json:
        predictor.selected_features = meta_json.get('feature_names', [
            "age","sex","sbp","dbp","hbp","duration","hb1ac","ldl","hdl","chol",
            "urea","bun","uric","egfr","trig","ucr","alt","ast",
            "age_hba1c_interaction","duration_egfr_ratio","ldl_hdl_ratio"
        ])
    else:
        predictor.selected_features = [
            "age","sex","sbp","dbp","hbp","duration","hb1ac","ldl","hdl","chol",
            "urea","bun","uric","egfr","trig","ucr","alt","ast",
            "age_hba1c_interaction","duration_egfr_ratio","ldl_hdl_ratio"
        ]
    
    # Set default thresholds
    predictor.tau0 = 0.5
    predictor.tau1 = 0.5
    
    # Handle different model types
    model_name = model_path.stem.lower()
    
    # Check if this is a dictionary-based ensemble model
    if isinstance(model_data, dict) and 'catboost_model' in model_data:
        # Handle dictionary-based ensemble model
        predictor.is_ensemble = True
        predictor.model_type = "StackedBlend"
        predictor.catboost_model = model_data['catboost_model']
        predictor.elasticnet_model = model_data['elasticnet_model']
        
        # Extract weights if available
        if 'weights' in model_data:
            weights = model_data['weights']
            predictor.weight_catboost = weights[0] if len(weights) > 0 else 0.6
            predictor.weight_elasticnet = weights[1] if len(weights) > 1 else 0.4
        else:
            predictor.weight_catboost = 0.6
            predictor.weight_elasticnet = 0.4
            
    elif "stacked" in model_name or "blend" in model_name:
        # Handle ensemble model with separate files
        predictor.is_ensemble = True
        predictor.model_type = "StackedBlend"
        
        # Try to load individual models for ensemble
        catboost_path = models_dir / "catboost_model.pkl"
        elasticnet_path = models_dir / "elasticnetlr_model.pkl"
        
        if catboost_path.exists() and elasticnet_path.exists():
            predictor.catboost_model = joblib.load(catboost_path)
            predictor.elasticnet_model = joblib.load(elasticnet_path)
            predictor.weight_catboost = 0.6  # Default weights
            predictor.weight_elasticnet = 0.4
        else:
            # Fall back to single model
            predictor.is_ensemble = False
            predictor.model = model_data
    else:
        # Single model
        predictor.is_ensemble = False
        predictor.model = model_data
        
        if "catboost" in model_name:
            predictor.model_type = "CatBoost"
        elif "lightgbm" in model_name:
            predictor.model_type = "LightGBM"
        elif "elasticnet" in model_name:
            predictor.model_type = "ElasticNet"
        elif "qda" in model_name:
            predictor.model_type = "QDA"
        else:
            predictor.model_type = "Unknown"
    
    return predictor

def list_available_models() -> List[str]:
    """List available predictor models"""
    if not models_dir.exists():
        return []
    
    # Look for actual model files in the directory
    model_files = []
    for pattern in ["*_model.pkl", "*_predictor.pkl"]:
        model_files.extend(models_dir.glob(pattern))
    
    return sorted([f.name for f in model_files])

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
    
    # Handle both _model.pkl and _predictor.pkl extensions
    if not (model_name.endswith("_model.pkl") or model_name.endswith("_predictor.pkl")):
        # Try both extensions
        model_path_1 = models_dir / f"{model_name}_model.pkl"
        model_path_2 = models_dir / f"{model_name}_predictor.pkl"
        
        if model_path_1.exists():
            model_path = model_path_1
            model_name = f"{model_name}_model.pkl"
        elif model_path_2.exists():
            model_path = model_path_2
            model_name = f"{model_name}_predictor.pkl"
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    else:
        model_path = models_dir / model_name
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        if model_name.endswith("_predictor.pkl"):
            # Load pre-wrapped predictor
            loaded_predictor = joblib.load(model_path)
        else:
            # Wrap raw model in predictor
            loaded_predictor = create_predictor_from_model(model_path)
        
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