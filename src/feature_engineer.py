"""
Feature engineering module for automatic encoding and imputation.
Handles categorical encoding, missing value imputation, and feature scaling.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging

from config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Automated feature engineering with intelligent preprocessing pipelines.
    
    Example:
        >>> engineer = FeatureEngineer()
        >>> X_processed = engineer.fit_transform(X_train, y_train)
        >>> engineer.save_pipeline("feature_pipeline.pkl")
    """
    
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        
    def create_pipeline(self, X: pd.DataFrame, problem_type: str = "classification") -> Pipeline:
        """
        Create preprocessing pipeline based on data characteristics.
        
        Args:
            X: Feature DataFrame
            problem_type: Type of ML problem
            
        Returns:
            Configured preprocessing pipeline
        """
        # Identify feature types
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(self.numerical_features)} numerical and "
                   f"{len(self.categorical_features)} categorical features")
        
        # Numerical preprocessing
        numerical_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])
        
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        return self.pipeline
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, problem_type: str = "classification") -> np.ndarray:
        """
        Fit preprocessing pipeline and transform features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            problem_type: Type of ML problem
            
        Returns:
            Transformed feature array
        """
        # Create and fit pipeline
        self.create_pipeline(X, problem_type)
        X_transformed = self.pipeline.fit_transform(X)
        
        # Store feature names for later use
        self._generate_feature_names()
        
        # Handle target encoding for classification
        if problem_type == "classification" and y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
            
        logger.info(f"Features transformed: {X.shape} -> {X_transformed.shape}")
        
        return X_transformed, y_encoded
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
            
        return self.pipeline.transform(X)
    
    def _generate_feature_names(self) -> None:
        """Generate feature names after transformation."""
        feature_names = []
        
        # Get transformer from pipeline
        preprocessor = self.pipeline.named_steps['preprocessor']
        
        # Numerical features (unchanged names)
        feature_names.extend(self.numerical_features)
        
        # Categorical features (get encoded names)
        if self.categorical_features:
            cat_transformer = preprocessor.named_transformers_['cat']
            encoder = cat_transformer.named_steps['encoder']
            
            try:
                encoded_names = encoder.get_feature_names_out(self.categorical_features)
                feature_names.extend(encoded_names)
            except AttributeError:
                # Fallback for older sklearn versions
                for cat_feature in self.categorical_features:
                    categories = encoder.categories_[self.categorical_features.index(cat_feature)]
                    for category in categories[1:]:  # Skip first due to drop='first'
                        feature_names.append(f"{cat_feature}_{category}")
        
        self.feature_names = feature_names
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        return self.feature_names
    
    def save_pipeline(self, filename: str) -> None:
        """Save preprocessing pipeline to disk."""
        if self.pipeline is None:
            raise ValueError("No pipeline to save")
            
        pipeline_path = MODELS_DIR / filename
        
        # Save pipeline and additional components
        save_dict = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }
        
        joblib.dump(save_dict, pipeline_path)
        logger.info(f"Pipeline saved to {pipeline_path}")
    
    def load_pipeline(self, filename: str) -> None:
        """Load preprocessing pipeline from disk."""
        pipeline_path = MODELS_DIR / filename
        
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        
        save_dict = joblib.load(pipeline_path)
        
        self.pipeline = save_dict['pipeline']
        self.label_encoder = save_dict.get('label_encoder')
        self.feature_names = save_dict.get('feature_names', [])
        self.categorical_features = save_dict.get('categorical_features', [])
        self.numerical_features = save_dict.get('numerical_features', [])
        
        logger.info(f"Pipeline loaded from {pipeline_path}")
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Inverse transform encoded target values."""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y_encoded)
        return y_encoded
    
    def get_feature_importance_mapping(self) -> Dict[str, int]:
        """Get mapping from feature names to indices."""
        return {name: idx for idx, name in enumerate(self.feature_names)}
    
    def create_time_series_features(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """
        Create time-based features for time series problems.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with additional time features
        """
        df_ts = df.copy()
        
        # Ensure datetime column is datetime type
        df_ts[datetime_col] = pd.to_datetime(df_ts[datetime_col])
        
        # Create time-based features
        df_ts['year'] = df_ts[datetime_col].dt.year
        df_ts['month'] = df_ts[datetime_col].dt.month
        df_ts['day'] = df_ts[datetime_col].dt.day
        df_ts['dayofweek'] = df_ts[datetime_col].dt.dayofweek
        df_ts['quarter'] = df_ts[datetime_col].dt.quarter
        df_ts['is_weekend'] = (df_ts[datetime_col].dt.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for seasonal features
        df_ts['month_sin'] = np.sin(2 * np.pi * df_ts['month'] / 12)
        df_ts['month_cos'] = np.cos(2 * np.pi * df_ts['month'] / 12)
        df_ts['day_sin'] = np.sin(2 * np.pi * df_ts['day'] / 31)
        df_ts['day_cos'] = np.cos(2 * np.pi * df_ts['day'] / 31)
        
        logger.info(f"Created time series features: {df.shape} -> {df_ts.shape}")
        
        return df_ts
