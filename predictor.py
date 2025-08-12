# predictor.py
import pickle
import json
import joblib
from typing import Dict, List, Tuple, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

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
        """Apply the same feature engineering as training"""
        df = df.copy()
        
        # Interaction 1: Age × HbA1c
        if 'age' in df.columns and 'hb1ac' in df.columns:
            df['age_hba1c_interaction'] = df['age'] * df['hb1ac']
        elif 'age' in df.columns and 'hba1c' in df.columns:
            df['age_hba1c_interaction'] = df['age'] * df['hba1c']
        
        # Interaction 2: Duration / eGFR
        if 'duration' in df.columns and 'egfr' in df.columns:
            df['duration_egfr_ratio'] = df['duration'] / (df['egfr'] + 1e-6)
        
        # Interaction 3: LDL / HDL ratio
        if 'ldl' in df.columns and 'hdl' in df.columns:
            df['ldl_hdl_ratio'] = df['ldl'] / (df['hdl'] + 1e-6)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _apply_temperature_scaling(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to probabilities"""
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
        """Apply optimized thresholds to convert probabilities to class predictions"""
        p0, p1, p2 = probs.T
        preds = np.where(p0 > self.tau0, 0, np.where(p1 > self.tau1, 1, 2))
        return preds
    
    def _preprocess_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing (scaling, encoding)"""
        X_processed = X.copy()
        
        # Apply scaling
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
        
        # Apply label encoding
        if self.label_encoders is not None:
            for feature, encoder in self.label_encoders.items():
                if feature in X_processed.columns:
                    try:
                        X_processed[feature] = encoder.transform(X_processed[feature])
                    except Exception as e:
                        print(f"Warning: Label encoding failed for {feature}: {e}")
        
        return X_processed
    
    def predict_proba(self, X: Union[pd.DataFrame, dict]) -> np.ndarray:
        """Generate probability predictions"""
        # Convert input to DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = X.to_frame().T
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame, Series, or dictionary")
        
        # Feature engineering
        X_processed = self._engineer_features(X)
        
        # Handle missing features
        if self.selected_features:
            missing_features = set(self.selected_features) - set(X_processed.columns)
            if missing_features:
                for feat in missing_features:
                    if self.feature_stats and feat in self.feature_stats.get('means', {}):
                        X_processed[feat] = self.feature_stats['means'][feat]
                    else:
                        X_processed[feat] = 0
            
            X_processed = X_processed[self.selected_features]
        
        # Fill missing values
        if self.feature_stats:
            for col in X_processed.columns:
                if col in self.feature_stats.get('means', {}):
                    X_processed[col] = X_processed[col].fillna(self.feature_stats['means'][col])
        
        X_processed = X_processed.fillna(0)
        
        # Apply preprocessing
        X_processed = self._preprocess_input(X_processed)
        
        # Generate predictions
        if self.is_ensemble:
            # Ensemble model
            catboost_probs = self.catboost_model.predict_proba(X_processed)
            elasticnet_probs = self.elasticnet_model.predict_proba(X_processed)
            
            # Apply temperature scaling
            catboost_probs = self._apply_temperature_scaling(catboost_probs, self.temperature_catboost)
            elasticnet_probs = self._apply_temperature_scaling(elasticnet_probs, self.temperature_elasticnet)
            
            # Weighted combination
            probs = self.weight_catboost * catboost_probs + self.weight_elasticnet * elasticnet_probs
        else:
            # Single model
            probs = self.model.predict_proba(X_processed)
            probs = self._apply_temperature_scaling(probs, self.temperature)
        
        return probs
    
    def predict(self, X: Union[pd.DataFrame, dict]) -> np.ndarray:
        """Generate class predictions"""
        probs = self.predict_proba(X)
        return self._apply_thresholds(probs)
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, dict]) -> Dict:
        """Generate predictions with confidence scores and risk assessment"""
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
        """Get feature importance if available"""
        if self.model_type == "CatBoost" and hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            return dict(zip(self.selected_features, importance))
        elif hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.selected_features, self.model.feature_importances_))
        else:
            return {}
