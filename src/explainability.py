"""
SHAP-based explainability module for global and local model interpretations.
Provides visual insights for any model type with unified interface.
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import joblib

from config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    SHAP-based model explainability with global and local interpretations.
    
    Example:
        >>> explainer = ExplainabilityEngine()
        >>> explainer.fit(model, X_train, feature_names)
        >>> global_importance = explainer.get_global_importance()
        >>> local_explanation = explainer.explain_instance(X_test.iloc[0])
    """
    
    def __init__(self):
        self.explainer: Optional[Any] = None
        self.shap_values: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.model: Optional[Any] = None
        self.X_background: Optional[np.ndarray] = None
        
    def fit(self, model: Any, X_background: np.ndarray, 
            feature_names: List[str], model_type: str = "auto") -> None:
        """
        Fit SHAP explainer to the model.
        
        Args:
            model: Trained model to explain
            X_background: Background dataset for SHAP
            feature_names: Names of features
            model_type: Type of model for explainer selection
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        
        # Select appropriate SHAP explainer
        try:
            if model_type == "auto":
                model_type = self._detect_model_type(model)
            
            if model_type in ["tree", "ensemble"]:
                self.explainer = shap.TreeExplainer(model)
                logger.info("Using TreeExplainer")
            elif model_type == "linear":
                self.explainer = shap.LinearExplainer(model, X_background)
                logger.info("Using LinearExplainer")
            elif model_type == "deep":
                self.explainer = shap.DeepExplainer(model, X_background)
                logger.info("Using DeepExplainer")
            else:
                # Default to KernelExplainer (model-agnostic)
                self.explainer = shap.KernelExplainer(model.predict, X_background)
                logger.info("Using KernelExplainer (model-agnostic)")
                
        except Exception as e:
            logger.warning(f"Error creating specific explainer: {e}")
            # Fallback to KernelExplainer
            self.explainer = shap.KernelExplainer(model.predict, X_background)
            logger.info("Fallback to KernelExplainer")
    
    def calculate_shap_values(self, X: np.ndarray, max_samples: int = 1000) -> np.ndarray:
        """
        Calculate SHAP values for given dataset.
        
        Args:
            X: Dataset to explain
            max_samples: Maximum samples to process (for performance)
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Limit samples for performance
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            logger.info(f"Sampling {max_samples} instances for SHAP calculation")
        else:
            X_sample = X
        
        try:
            self.shap_values = self.explainer.shap_values(X_sample)
            logger.info(f"SHAP values calculated for {len(X_sample)} samples")
            return self.shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            raise
    
    def get_global_importance(self, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get global feature importance based on mean absolute SHAP values.
        
        Args:
            X: Optional dataset to calculate SHAP values
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.shap_values is None:
            if X is not None:
                self.calculate_shap_values(X)
            else:
                raise ValueError("No SHAP values available. Provide X or call calculate_shap_values first.")
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            # Average across classes
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
        else:
            mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create importance dictionary
        importance_dict = {}
        for i, importance in enumerate(mean_shap):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            importance_dict[feature_name] = float(importance)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def explain_instance(self, X_instance: np.ndarray, 
                        show_plot: bool = False) -> Dict[str, Any]:
        """
        Generate local explanation for a single instance.
        
        Args:
            X_instance: Single instance to explain
            show_plot: Whether to display force plot
            
        Returns:
            Local explanation dictionary
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Ensure instance is 2D
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Calculate SHAP values for instance
        instance_shap = self.explainer.shap_values(X_instance)
        
        # Handle multi-class case
        if isinstance(instance_shap, list):
            # Use first class for simplicity
            instance_shap = instance_shap[0]
        
        # Create explanation dictionary
        explanation = {
            "shap_values": instance_shap[0].tolist(),
            "feature_contributions": {},
            "base_value": getattr(self.explainer, 'expected_value', 0.0),
            "prediction": self.model.predict(X_instance)[0] if self.model else None
        }
        
        # Feature contributions
        for i, shap_val in enumerate(instance_shap[0]):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            explanation["feature_contributions"][feature_name] = {
                "value": float(X_instance[0, i]),
                "shap_value": float(shap_val),
                "contribution": "positive" if shap_val > 0 else "negative"
            }
        
        # Generate force plot if requested
        if show_plot:
            try:
                shap.force_plot(
                    self.explainer.expected_value,
                    instance_shap[0],
                    X_instance[0],
                    feature_names=self.feature_names,
                    show=True
                )
            except Exception as e:
                logger.warning(f"Could not generate force plot: {e}")
        
        return explanation
    
    def generate_summary_plot(self, X: Optional[np.ndarray] = None, 
                            save_path: Optional[str] = None) -> None:
        """
        Generate and save SHAP summary plot.
        
        Args:
            X: Dataset for plotting (uses background if None)
            save_path: Path to save plot
        """
        if self.shap_values is None:
            if X is not None:
                self.calculate_shap_values(X)
            else:
                raise ValueError("No SHAP values available")
        
        plt.figure(figsize=(10, 6))
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            shap.summary_plot(self.shap_values[0], self.X_background, 
                            feature_names=self.feature_names, show=False)
        else:
            shap.summary_plot(self.shap_values, self.X_background,
                            feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Summary plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_waterfall_plot(self, instance_idx: int, X: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Generate waterfall plot for a specific instance.
        
        Args:
            instance_idx: Index of instance to plot
            X: Dataset containing the instance
            save_path: Path to save plot
        """
        if X is None:
            X = self.X_background
            
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        try:
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[instance_idx],
                    base_values=getattr(self.explainer, 'expected_value', 0.0),
                    data=X[instance_idx],
                    feature_names=self.feature_names
                ),
                show=False
            )
        except Exception as e:
            logger.warning(f"Could not generate waterfall plot: {e}")
            return
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Waterfall plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_top_features(self, n_features: int = 10, 
                        X: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """
        Get top N most important features globally.
        
        Args:
            n_features: Number of top features to return
            X: Optional dataset for SHAP calculation
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        importance_dict = self.get_global_importance(X)
        return list(importance_dict.items())[:n_features]
    
    def save_explainer(self, filename: str) -> None:
        """Save fitted explainer to disk."""
        if self.explainer is None:
            raise ValueError("No explainer to save")
        
        explainer_path = MODELS_DIR / filename
        
        explainer_package = {
            'explainer': self.explainer,
            'shap_values': self.shap_values,
            'feature_names': self.feature_names,
            'X_background': self.X_background
        }
        
        joblib.dump(explainer_package, explainer_path)
        logger.info(f"Explainer saved to {explainer_path}")
    
    def load_explainer(self, filename: str) -> None:
        """Load fitted explainer from disk."""
        explainer_path = MODELS_DIR / filename
        
        if not explainer_path.exists():
            raise FileNotFoundError(f"Explainer file not found: {explainer_path}")
        
        explainer_package = joblib.load(explainer_path)
        
        self.explainer = explainer_package['explainer']
        self.shap_values = explainer_package.get('shap_values')
        self.feature_names = explainer_package.get('feature_names', [])
        self.X_background = explainer_package.get('X_background')
        
        logger.info(f"Explainer loaded from {explainer_path}")
    
    def _detect_model_type(self, model: Any) -> str:
        """Detect model type for appropriate explainer selection."""
        model_name = type(model).__name__.lower()
        
        if any(tree_name in model_name for tree_name in 
               ['tree', 'forest', 'boost', 'xgb', 'lgb', 'catboost']):
            return "tree"
        elif any(linear_name in model_name for linear_name in 
                ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
            return "linear"
        elif any(deep_name in model_name for deep_name in 
                ['neural', 'mlp', 'deep', 'keras', 'torch']):
            return "deep"
        else:
            return "unknown"
