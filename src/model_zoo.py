"""
Model zoo containing implementations of various ML algorithms.
Provides unified interface for ElasticNet, RF, GB, XGB, LSTM, and TabNet.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
# Temporarily disable TensorFlow imports to avoid DLL issues
# import tensorflow as tf
# from tensorflow import keras
# from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import logging

from config import RANDOM_SEED, MAX_ITER

# Avoid importing TensorFlow at import-time to prevent Windows DLL issues during testing
keras = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelZoo:
    """
    Centralized model repository with unified interface.
    
    Example:
        >>> zoo = ModelZoo()
        >>> model = zoo.get_model("xgboost", "classification")
        >>> model.fit(X_train, y_train)
    """
    
    def __init__(self):
        self.models = {
            "elastic_net": self._create_elastic_net,
            "random_forest": self._create_random_forest,
            "gradient_boosting": self._create_gradient_boosting,
            "xgboost": self._create_xgboost,
            # Optional/conditional model (requires TensorFlow):
            "lstm": self._create_lstm,
            "linear_regression": self._create_linear_regression,
            # "tabnet": self._create_tabnet,
            # "prophet": self._create_prophet
        }
    
    def get_model(self, model_name: str, problem_type: str) -> Any:
        """
        Get model instance by name and problem type.
        
        Args:
            model_name: Name of the model
            problem_type: 'classification', 'regression', or 'time_series'
            
        Returns:
            Configured model instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in zoo")
        
        return self.models[model_name](problem_type)
    
    def get_available_models(self) -> list:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def _create_elastic_net(self, problem_type: str) -> Any:
        """Create ElasticNet model (regression only)."""
        if problem_type == "classification":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=0.5,
                max_iter=MAX_ITER,
                random_state=RANDOM_SEED
            )
        else:
            return ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                max_iter=MAX_ITER,
                random_state=RANDOM_SEED
            )
    
    def _create_random_forest(self, problem_type: str) -> Any:
        """Create Random Forest model."""
        if problem_type == "classification":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
    
    def _create_gradient_boosting(self, problem_type: str) -> Any:
        """Create Gradient Boosting model."""
        if problem_type == "classification":
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_SEED
            )
        else:
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_SEED
            )
    
    def _create_xgboost(self, problem_type: str) -> Any:
        """Create XGBoost model."""
        if problem_type == "classification":
            return xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_state=RANDOM_SEED,
                eval_metric='logloss'
            )
        else:
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_SEED
            )

    def _create_linear_regression(self, problem_type: str) -> Any:
        """Create Linear Regression model (regression only)."""
        if problem_type != "regression":
            raise ValueError("Linear Regression is only applicable to regression problems")
        return LinearRegression()
    
    def _create_lstm(self, problem_type: str) -> Any:
        """Create LSTM model for time series.

        Note: Imports TensorFlow/Keras lazily to avoid DLL issues on Windows
        when running environments that don't have TF installed.
        """
        # Lazy-import keras only when LSTM is requested
        global keras  # use module-level name referenced by LSTMModel
        if keras is None:  # type: ignore[name-defined]
            try:  # pragma: no cover - environment dependent
                from tensorflow import keras as _keras  # type: ignore
                keras = _keras  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise NotImplementedError(
                    "LSTM requires TensorFlow/Keras which is not available in this environment."
                ) from exc
        return LSTMModel(problem_type)
    
    def _create_tabnet(self, problem_type: str) -> Any:
        """Create TabNet model."""
        raise NotImplementedError("TabNet temporarily disabled due to TensorFlow issues")
    
    def _create_prophet(self, problem_type: str) -> Any:
        """Create Prophet model for time series."""
        raise NotImplementedError("Prophet temporarily disabled")


class LSTMModel:
    """
    LSTM wrapper for time series prediction with sklearn-like interface.
    """
    
    def __init__(self, problem_type: str, sequence_length: int = 10):
        self.problem_type = problem_type
        self.sequence_length = sequence_length
        # When keras is unavailable, surface a clear error upon usage
        if keras is None:
            # Delay error until someone actually tries to instantiate this
            # class via ModelZoo; provide a clear message.
            raise NotImplementedError(
                "LSTM requires TensorFlow/Keras which is not available in this environment."
            )
        self.model: Optional[keras.Model] = None  # type: ignore[name-defined]
        self.scaler = None
        
    def _create_sequences(self, data: np.ndarray) -> tuple:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSTMModel':
        """Fit LSTM model."""
        # Prepare sequences
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(np.column_stack([X, y]))
        
        # Build model
        self.model = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True, input_shape=(self.sequence_length, X.shape[1] + 1)),
            keras.layers.LSTM(50),
            keras.layers.Dense(25),
            keras.layers.Dense(1 if self.problem_type == "regression" else len(np.unique(y)))
        ])
        
        # Compile model
        if self.problem_type == "classification":
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        self.model.fit(X_seq[:, :, :-1], y_seq, epochs=50, batch_size=32, verbose=0)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # For prediction, use last sequence_length points
        if len(X) >= self.sequence_length:
            X_pred = X[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            return self.model.predict(X_pred, verbose=0).flatten()
        else:
            # Handle case where we don't have enough data
            return np.zeros(len(X))
