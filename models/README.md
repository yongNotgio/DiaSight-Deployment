# Diabetes Risk Prediction Models

This directory contains trained models for diabetic retinopathy risk prediction.

## Files:
- `*_predictor.pkl`: Main model files (use these)
- `*_predictor.pickle`: Backup model files
- `model_metadata.json`: Complete model information
- `feature_statistics.json`: Training data statistics
- `model_loader.py`: Utility functions for loading models

## Quick Start:
```python
import joblib

# Load the recommended model
model = joblib.load("models/catboost_predictor.pkl")

# Make prediction
patient_data = {
    "age": 65, "hb1ac": 8.5, "duration": 10, 
    "egfr": 60, "ldl": 120, "hdl": 40, "chol": 200
}

result = model.predict_with_confidence(patient_data)
print(f"Prediction: {result['class_names'][result['predictions'][0]]}")
print(f"Confidence: {result['confidence'][0]:.1%}")
print(f"Risk Score: {result['risk_scores'][0]:.1f}%")
```

## Models Available:
- **catboost**: Best single model (recommended)
- **stackedblend**: Ensemble model (highest accuracy)
- **elasticnetlr**: Linear model (interpretable)
- **qda**: Quadratic discriminant analysis

## Requirements:
- numpy, pandas, scikit-learn, catboost
