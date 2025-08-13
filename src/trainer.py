"""
Unified training interface for all models with validation and warm-start support.
Provides consistent training pipeline across different algorithms.
"""
import joblib
import json
import logging
from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import cross_val_score

from config import MODELS_DIR, TRAINING_LOG_FILE, CV_FOLDS, RANDOM_SEED, TRAIN_TOP_K
from meta_engine import MetaEngine
from feature_engineer import FeatureEngineer
from model_zoo import ModelZoo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Unified training interface with validation and persistence.
    
    Example:
        >>> trainer = Trainer()
        >>> results = trainer.train_best_model(X_train, y_train, X_test, y_test, "classification")
        >>> trainer.save_model("best_model.pkl")
    """
    
    def __init__(self):
        self.meta_engine = MetaEngine()
        self.feature_engineer = FeatureEngineer()
        self.model_zoo = ModelZoo()
        self.best_model: Optional[Any] = None
        self.best_model_name: str = ""
        self.best_score: float = 0.0
        self.training_history: list = []
        
    def train_best_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        problem_type: str) -> Dict[str, Any]:
        """
        Complete training pipeline: feature engineering + model selection + training.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features  
            y_test: Test target
            problem_type: Type of ML problem
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting complete training pipeline")
        
        # Feature engineering
        X_train_processed, y_train_processed = self.feature_engineer.fit_transform(
            X_train, y_train, problem_type
        )
        X_test_processed = self.feature_engineer.transform(X_test)
        
        # Get model recommendations - force the correct problem type
        analysis = self.meta_engine.analyze_and_recommend(X_train, y_train, problem_type)
        # Override the problem type in the analysis to ensure consistency
        analysis["dataset_profile"]["problem_type"] = problem_type
        # Also override in the recommendations to ensure consistency
        for i, (model_name, confidence) in enumerate(analysis["model_recommendations"]):
            if problem_type == "classification":
                # Ensure we're using classification models
                if model_name in ["elastic_net"]:
                    analysis["model_recommendations"][i] = ("random_forest", confidence)
        recommendations = analysis["model_recommendations"]
        
        # Train and evaluate multiple models
        results = {
            "models_tested": [],
            "best_model_name": "",
            "best_score": 0.0,
            "best_params": {},
            "training_scores": {},
            "test_scores": {},
            "cross_val_scores": {},
            "feature_importance": {},
            "analysis": analysis
        }
        
        best_score = -np.inf if problem_type == "regression" else 0.0
        
        # Test top-K recommended models (configurable)
        for model_name, confidence in recommendations[:TRAIN_TOP_K]:
            logger.info(f"Training {model_name} (confidence: {confidence:.2f})")
            
            try:
                # Optimize hyperparameters
                model, best_params, cv_score = self.meta_engine.optimize_hyperparameters(
                    model_name, X_train_processed, y_train_processed, problem_type
                )
                
                # Evaluate on test set
                test_score = self._evaluate_model(
                    model, X_test_processed, y_test, problem_type
                )
                
                # Store results
                results["models_tested"].append(model_name)
                results["training_scores"][model_name] = cv_score
                results["test_scores"][model_name] = test_score
                results["cross_val_scores"][model_name] = cv_score
                
                # Check if this is the best model
                score_to_compare = test_score if problem_type != "regression" else -test_score
                if score_to_compare > best_score:
                    best_score = score_to_compare
                    self.best_model = model
                    self.best_model_name = model_name
                    self.best_score = test_score
                    results["best_model_name"] = model_name
                    results["best_score"] = test_score
                    results["best_params"] = best_params
                
                logger.info(f"{model_name} - CV: {cv_score:.4f}, Test: {test_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Generate feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self.feature_engineer.get_feature_names()
            if len(feature_names) == len(self.best_model.feature_importances_):
                importance_dict = dict(zip(feature_names, self.best_model.feature_importances_))
                results["feature_importance"] = dict(sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                )[:20])  # Top 20 features
        
        # Save training results
        self._log_training_results(results)
        
        logger.info(f"Training complete. Best model: {self.best_model_name} "
                   f"(score: {self.best_score:.4f})")
        
        return results
    
    def warm_start_training(self, X_new: pd.DataFrame, y_new: pd.Series,
                           problem_type: str) -> Dict[str, Any]:
        """
        Continue training existing model with new data (warm start).
        
        Args:
            X_new: New training features
            y_new: New training target
            problem_type: Type of ML problem
            
        Returns:
            Updated training results
        """
        if self.best_model is None:
            raise ValueError("No existing model to warm start. Train a model first.")
        
        logger.info(f"Warm starting {self.best_model_name} with {len(X_new)} new samples")
        
        # Process new features using existing pipeline
        X_new_processed = self.feature_engineer.transform(X_new)
        
        # Check if model supports warm start
        if hasattr(self.best_model, 'partial_fit'):
            # Incremental learning
            self.best_model.partial_fit(X_new_processed, y_new)
        elif hasattr(self.best_model, 'warm_start') and self.best_model_name in ['gradient_boosting']:
            # Warm start for ensemble methods
            self.best_model.set_params(warm_start=True, n_estimators=self.best_model.n_estimators + 50)
            
            # Combine with previous data (simplified approach)
            # In production, you'd want to maintain a sliding window
            self.best_model.fit(X_new_processed, y_new)
        else:
            # Full retraining
            logger.warning(f"{self.best_model_name} doesn't support warm start, performing full retrain")
            self.best_model.fit(X_new_processed, y_new)
        
        # Evaluate updated model
        new_score = self._evaluate_model(self.best_model, X_new_processed, y_new, problem_type)
        
        results = {
            "model_name": self.best_model_name,
            "new_samples": len(X_new),
            "updated_score": new_score,
            "previous_score": self.best_score,
            "improvement": new_score - self.best_score
        }
        
        self.best_score = new_score
        
        logger.info(f"Warm start complete. Updated score: {new_score:.4f} "
                   f"(improvement: {results['improvement']:+.4f})")
        
        return results
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                       problem_type: str) -> float:
        """Evaluate model performance with appropriate metrics.

        Prefer ROC-AUC for classification when probability scores are available; else accuracy.
        """
        # Classification
        if problem_type == "classification":
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    # Handle binary vs multi-class
                    if proba.shape[1] == 2:
                        from sklearn.metrics import roc_auc_score
                        return roc_auc_score(y, proba[:, 1])
                    else:
                        from sklearn.metrics import roc_auc_score
                        return roc_auc_score(y, proba, multi_class="ovr", average="weighted")
            except Exception:
                pass
            # Fallback
            predictions = model.predict(X)
            return accuracy_score(y, predictions)
        
        # Regression
        predictions = model.predict(X)
        return r2_score(y, predictions)
    
    def _get_detailed_metrics(self, model: Any, X: np.ndarray, y: np.ndarray,
                            problem_type: str) -> Dict[str, Any]:
        """Get comprehensive evaluation metrics."""
        predictions = model.predict(X)
        
        if problem_type == "classification":
            return {
                "accuracy": accuracy_score(y, predictions),
                "classification_report": classification_report(y, predictions, output_dict=True)
            }
        else:
            return {
                "r2_score": r2_score(y, predictions),
                "mse": mean_squared_error(y, predictions),
                "rmse": np.sqrt(mean_squared_error(y, predictions))
            }
    
    def save_model(self, filename: str) -> None:
        """Save trained model and associated components."""
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        model_path = MODELS_DIR / filename
        
        # Save complete model package
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'score': self.best_score,
            'feature_engineer': self.feature_engineer,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        joblib.dump(model_package, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename: str) -> None:
        """Load trained model and associated components."""
        model_path = MODELS_DIR / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_package = joblib.load(model_path)
        
        self.best_model = model_package['model']
        self.best_model_name = model_package['model_name']
        self.best_score = model_package['score']
        self.feature_engineer = model_package['feature_engineer']
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model.

        For classifiers, this method will prefer returning probabilities of the
        positive class when available, to support probability-first UIs.
        """
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        # Process features
        X_processed = self.feature_engineer.transform(X)
        
        # Prefer probabilities when the model supports it
        try:
            if hasattr(self.best_model, 'predict_proba'):
                proba = self.best_model.predict_proba(X_processed)
                # Binary classification → return prob of class 1
                if proba.ndim == 2 and proba.shape[1] == 2:
                    return proba[:, 1]
                return proba
        except Exception:
            pass

        # Fallback: derive probabilities from decision function if possible
        try:
            if hasattr(self.best_model, 'decision_function'):
                scores = self.best_model.decision_function(X_processed)
                # Binary case: sigmoid
                import numpy as _np
                if scores.ndim == 1:
                    return 1.0 / (1.0 + _np.exp(-scores))
                # Multi-class: softmax
                exps = _np.exp(scores - _np.max(scores, axis=1, keepdims=True))
                return exps / _np.sum(exps, axis=1, keepdims=True)
        except Exception:
            pass

        # Final fallback to raw predictions (e.g., regression models)
        predictions = self.best_model.predict(X_processed)
        
        # Inverse transform if needed
        if hasattr(self.feature_engineer, 'label_encoder') and self.feature_engineer.label_encoder:
            predictions = self.feature_engineer.inverse_transform_target(predictions)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities for classification."""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        # Process features
        X_processed = self.feature_engineer.transform(X)

        # Native probabilities when available
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X_processed)

        # Fallback: construct probabilities from decision_function
        if hasattr(self.best_model, 'decision_function'):
            scores = self.best_model.decision_function(X_processed)
            import numpy as _np
            if scores.ndim == 1:
                # Binary: sigmoid to [0,1]; return two-column probs for consistency
                p1 = 1.0 / (1.0 + _np.exp(-scores))
                return _np.vstack([1.0 - p1, p1]).T
            # Multi-class: softmax rows
            exps = _np.exp(scores - _np.max(scores, axis=1, keepdims=True))
            return exps / _np.sum(exps, axis=1, keepdims=True)

        raise ValueError("Model doesn't support probability predictions")
    
    def _log_training_results(self, results: Dict[str, Any]) -> None:
        """Log training results to file."""
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "results": results
        }
        
        log_path = MODELS_DIR / TRAINING_LOG_FILE
        
        # Load existing log or create new
        if log_path.exists():
            with open(log_path, 'r') as f:
                training_log = json.load(f)
        else:
            training_log = []
        
        # Add new entry
        training_log.append(log_entry)
        
        # Save updated log
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        logger.info(f"Training results logged to {log_path}")
