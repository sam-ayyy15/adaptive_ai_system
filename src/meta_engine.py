"""
Meta-engine for intelligent model selection and hyperparameter optimization.
Provides data-driven algorithm recommendation and automated tuning.
"""
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

from config import (
    MODELS_DIR, SMALL_DATASET_ROWS, SMALL_DATASET_FEATURES,
    CV_FOLDS, RANDOM_SEED, MODEL_CONFIGS, METADATA_FILE
)
from model_zoo import ModelZoo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaEngine:
    """
    Intelligent meta-learning engine for automatic model selection and optimization.
    
    Example:
        >>> engine = MetaEngine()
        >>> recommendations = engine.analyze_and_recommend(X, y, problem_type)
        >>> best_model = engine.optimize_model(model_name, X, y)
    """
    
    def __init__(self):
        self.model_zoo = ModelZoo()
        self.metadata: Dict[str, Any] = {}
        self.dataset_profile: Dict[str, Any] = {}
        
    def profile_dataset(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> Dict[str, Any]:
        """
        Profile dataset characteristics for intelligent model selection.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            problem_type: Type of ML problem
            
        Returns:
            Dataset profile dictionary
        """
        profile = {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "problem_type": problem_type,
            "target_distribution": {},
            "feature_types": {},
            "sparsity": {},
            "complexity_metrics": {}
        }
        
        # Target analysis
        if problem_type == "classification":
            profile["target_distribution"] = y.value_counts().to_dict()
            profile["n_classes"] = y.nunique()
            profile["is_balanced"] = self._check_class_balance(y)
        else:
            profile["target_distribution"] = {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max())
            }
        
        # Feature analysis
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        profile["feature_types"] = {
            "numeric": len(numeric_features),
            "categorical": len(categorical_features),
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features)
        }
        
        # Sparsity analysis
        profile["sparsity"] = {
            "missing_ratio": X.isnull().sum().sum() / (X.shape[0] * X.shape[1]),
            "zero_ratio": (X == 0).sum().sum() / (X.shape[0] * X.shape[1]) if len(numeric_features) > 0 else 0
        }
        
        # Complexity metrics
        profile["complexity_metrics"] = {
            "samples_per_feature": len(X) / len(X.columns) if len(X.columns) > 0 else 0,
            "is_high_dimensional": len(X.columns) > len(X) * 0.1,
            "is_small_dataset": len(X) < SMALL_DATASET_ROWS,
            "is_wide_dataset": len(X.columns) > SMALL_DATASET_FEATURES
        }
        
        self.dataset_profile = profile
        logger.info(f"Dataset profiled: {profile['n_samples']} samples, "
                   f"{profile['n_features']} features, {problem_type}")
        
        return profile
    
    def recommend_models(self, dataset_profile: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Recommend models based on dataset characteristics.
        
        Args:
            dataset_profile: Dataset profile from profile_dataset()
            
        Returns:
            List of (model_name, confidence_score) tuples
        """
        recommendations = []
        
        n_samples = dataset_profile["n_samples"]
        n_features = dataset_profile["n_features"]
        problem_type = dataset_profile["problem_type"]
        is_balanced = dataset_profile.get("is_balanced", True)
        
        # Time series detection
        if problem_type == "time_series":
            recommendations.extend([
                ("lstm", 0.9),
                ("prophet", 0.8)
            ])
            return recommendations
        
        # Small dataset recommendations
        if n_samples < SMALL_DATASET_ROWS:
            if problem_type == "classification":
                recommendations.extend([
                    ("random_forest", 0.8),
                    ("gradient_boosting", 0.7),
                    ("elastic_net", 0.6)
                ])
            else:
                recommendations.extend([
                    ("random_forest", 0.8),
                    ("elastic_net", 0.7),
                    ("gradient_boosting", 0.6)
                ])
        
        # Large dataset recommendations
        else:
            if problem_type == "classification":
                recommendations.extend([
                    ("xgboost", 0.9),
                    ("gradient_boosting", 0.8),
                    ("random_forest", 0.7)
                ])
                
                # Add deep learning for very large datasets
                if n_samples > 50000:
                    recommendations.append(("tabnet", 0.8))
            else:
                recommendations.extend([
                    ("xgboost", 0.9),
                    ("gradient_boosting", 0.8),
                    ("elastic_net", 0.7),
                    ("random_forest", 0.6)
                ])
        
        # High-dimensional data adjustments
        if n_features > n_samples * 0.1:
            # Boost regularized models
            recommendations = [(name, score + 0.1 if name in ["elastic_net"] else score) 
                             for name, score in recommendations]
        
        # Imbalanced data adjustments
        if not is_balanced and problem_type == "classification":
            # Boost ensemble methods
            recommendations = [(name, score + 0.1 if name in ["random_forest", "gradient_boosting", "xgboost"] else score) 
                             for name, score in recommendations]
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Model recommendations: {recommendations[:3]}")
        return recommendations
    
    def optimize_hyperparameters(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                                problem_type: str) -> Tuple[Any, Dict[str, Any], float]:
        """
        Optimize hyperparameters using grid search or randomized search.
        
        Args:
            model_name: Name of model to optimize
            X: Feature array
            y: Target array
            problem_type: Type of ML problem
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        # Get base model and parameter grid
        base_model = self.model_zoo.get_model(model_name, problem_type)
        param_grid = MODEL_CONFIGS.get(model_name, {}).get("param_grid", {})
        
        if not param_grid:
            logger.warning(f"No parameter grid found for {model_name}, using default parameters")
            base_model.fit(X, y)
            return base_model, {}, self._evaluate_model(base_model, X, y, problem_type)
        
        # Choose search strategy based on dataset size
        n_samples = len(X)
        
        if n_samples < SMALL_DATASET_ROWS:
            # Use GridSearchCV for small datasets
            logger.info(f"Using GridSearchCV for {model_name}")
            search = GridSearchCV(
                base_model, param_grid, cv=CV_FOLDS,
                scoring=self._get_scoring_metric(problem_type),
                n_jobs=-1
            )
        else:
            # Use RandomizedSearchCV for large datasets
            logger.info(f"Using RandomizedSearchCV for {model_name}")
            search = RandomizedSearchCV(
                base_model, param_grid, cv=CV_FOLDS,
                scoring=self._get_scoring_metric(problem_type),
                n_jobs=-1, random_state=RANDOM_SEED,
                n_iter=20  # Limit iterations for speed
            )
        
        # Fit and get best model
        search.fit(X, y)
        
        logger.info(f"Best {model_name} score: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def analyze_and_recommend(self, X: pd.DataFrame, y: pd.Series, 
                            problem_type: str) -> Dict[str, Any]:
        """
        Complete analysis and model recommendation pipeline.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            problem_type: Type of ML problem
            
        Returns:
            Complete analysis and recommendations
        """
        # Profile dataset
        profile = self.profile_dataset(X, y, problem_type)
        
        # Get model recommendations
        recommendations = self.recommend_models(profile)
        
        # Prepare analysis result
        analysis = {
            "dataset_profile": profile,
            "model_recommendations": recommendations,
            "recommended_strategy": self._get_recommended_strategy(profile),
            "expected_challenges": self._identify_challenges(profile)
        }
        
        return analysis
    
    def _check_class_balance(self, y: pd.Series) -> bool:
        """Check if classification target is balanced."""
        value_counts = y.value_counts()
        min_class_ratio = value_counts.min() / value_counts.max()
        return min_class_ratio > 0.3  # Consider balanced if ratio > 30%
    
    def _get_scoring_metric(self, problem_type: str) -> str:
        """Get appropriate scoring metric for problem type."""
        if problem_type == "classification":
            return "accuracy"
        else:
            return "neg_mean_squared_error"
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                       problem_type: str) -> float:
        """Evaluate model performance."""
        predictions = model.predict(X)
        
        if problem_type == "classification":
            return accuracy_score(y, predictions)
        else:
            return r2_score(y, predictions)
    
    def _get_recommended_strategy(self, profile: Dict[str, Any]) -> str:
        """Get recommended training strategy based on profile."""
        n_samples = profile["n_samples"]
        n_features = profile["n_features"]
        
        if n_samples < SMALL_DATASET_ROWS and n_features < SMALL_DATASET_FEATURES:
            return "exhaustive_grid_search"
        elif n_samples > 100000:
            return "randomized_search_with_early_stopping"
        else:
            return "randomized_search_with_cross_validation"
    
    def _identify_challenges(self, profile: Dict[str, Any]) -> List[str]:
        """Identify potential challenges based on dataset profile."""
        challenges = []
        
        if profile["sparsity"]["missing_ratio"] > 0.2:
            challenges.append("high_missing_values")
        
        if profile["complexity_metrics"]["is_high_dimensional"]:
            challenges.append("high_dimensionality")
        
        if profile["problem_type"] == "classification" and not profile.get("is_balanced", True):
            challenges.append("class_imbalance")
        
        if profile["n_samples"] < 1000:
            challenges.append("small_sample_size")
        
        return challenges
    
    def save_metadata(self, model_name: str, best_params: Dict[str, Any], 
                     best_score: float, dataset_profile: Dict[str, Any]) -> None:
        """Save model metadata to disk."""
        metadata = {
            "model_name": model_name,
            "best_parameters": best_params,
            "best_score": best_score,
            "dataset_profile": dataset_profile,
            "timestamp": pd.Timestamp.now().isoformat(),
            "random_seed": RANDOM_SEED
        }
        
        metadata_path = MODELS_DIR / METADATA_FILE
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from disk."""
        metadata_path = MODELS_DIR / METADATA_FILE
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        return self.metadata
