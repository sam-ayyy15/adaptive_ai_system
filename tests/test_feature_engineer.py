### tests/test_feature_engineer.py
"""
Tests for feature_engineer module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()
        
        # Create sample data
        self.X = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical1': ['A', 'B', 'A', 'C', 'B'],
            'categorical2': ['X', 'Y', 'X', 'X', 'Y']
        })
        
        self.y = pd.Series([0, 1, 0, 1, 1])
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        pipeline = self.engineer.create_pipeline(self.X, 'classification')
        
        assert pipeline is not None
        assert len(self.engineer.numerical_features) == 2
        assert len(self.engineer.categorical_features) == 2
    
    def test_fit_transform(self):
        """Test fitting and transforming data."""
        X_transformed, y_encoded = self.engineer.fit_transform(self.X, self.y, 'classification')
        
        assert X_transformed.shape[0] == len(self.X)
        assert X_transformed.shape[1] > len(self.X.columns)  # Due to one-hot encoding
        assert len(y_encoded) == len(self.y)
    
    def test_transform(self):
        """Test transforming new data."""
        # First fit
        self.engineer.fit_transform(self.X, self.y, 'classification')
        
        # Then transform new data
        X_new = self.X.copy()
        X_transformed = self.engineer.transform(X_new)
        
        assert X_transformed.shape[1] == len(self.engineer.get_feature_names())
    
    def test_feature_names_generation(self):
        """Test feature name generation."""
        self.engineer.fit_transform(self.X, self.y, 'classification')
        feature_names = self.engineer.get_feature_names()
        
        assert len(feature_names) > 0
        assert 'numeric1' in feature_names
        assert 'numeric2' in feature_names
