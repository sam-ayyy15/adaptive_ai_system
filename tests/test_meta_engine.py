### tests/test_meta_engine.py

"""
Tests for meta_engine module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meta_engine import MetaEngine


class TestMetaEngine:
    """Test cases for MetaEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MetaEngine()
        
        # Create sample data
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        self.y_classification = pd.Series(np.random.choice([0, 1], 100))
        self.y_regression = pd.Series(np.random.randn(100))
    
    def test_profile_dataset_classification(self):
        """Test dataset profiling for classification."""
        profile = self.engine.profile_dataset(self.X, self.y_classification, 'classification')
        
        assert profile['n_samples'] == 100
        assert profile['n_features'] == 3
        assert profile['problem_type'] == 'classification'
        assert 'target_distribution' in profile
        assert 'feature_types' in profile
    
    def test_profile_dataset_regression(self):
        """Test dataset profiling for regression."""
        profile = self.engine.profile_dataset(self.X, self.y_regression, 'regression')
        
        assert profile['problem_type'] == 'regression'
        assert 'mean' in profile['target_distribution']
        assert 'std' in profile['target_distribution']
    
    def test_recommend_models(self):
        """Test model recommendation."""
        profile = self.engine.profile_dataset(self.X, self.y_classification, 'classification')
        recommendations = self.engine.recommend_models(profile)
        
        assert len(recommendations) > 0
        assert all(isinstance(score, float) for _, score in recommendations)
        assert all(0 <= score <= 1 for _, score in recommendations)
    
    def test_analyze_and_recommend(self):
        """Test complete analysis and recommendation."""
        analysis = self.engine.analyze_and_recommend(self.X, self.y_classification, 'classification')
        
        assert 'dataset_profile' in analysis
        assert 'model_recommendations' in analysis
        assert 'recommended_strategy' in analysis
        assert 'expected_challenges' in analysis
