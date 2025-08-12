
# Model Loader Utility
import joblib
from pathlib import Path

def load_model(model_name="catboost"):
    """Load a trained model for prediction"""
    models_dir = Path("models")
    model_file = models_dir / f"{model_name}_predictor.pkl"

    if model_file.exists():
        return joblib.load(model_file)
    else:
        raise FileNotFoundError(f"Model {model_name} not found in {models_dir}")

def list_available_models():
    """List all available models"""
    models_dir = Path("models")
    models = [f.stem.replace("_predictor", "") for f in models_dir.glob("*_predictor.pkl")]
    return models

# Example usage:
# model = load_model("catboost")
# result = model.predict_with_confidence({"age": 65, "hb1ac": 8.5, "duration": 10, "egfr": 60, "ldl": 120, "hdl": 40, "chol": 200})
