### tests/test_trainer.py

"""
Tests for trainer module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from sklearn.datasets import make_classification

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trainer import Trainer


class TestTrainer:
    """Test cases for Trainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = Trainer()
        
        # Create sample dataset
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        self.X_train = pd.DataFrame(X[:80], columns=[f'feature_{i}' for i in range(10)])
        self.X_test = pd.DataFrame(X[80:], columns=[f'feature_{i}' for i in range(10)])
        self.y_train = pd.Series(y[:80])
        self.y_test = pd.Series(y[80:])
    
    def test_train_best_model(self):
        """Test complete training pipeline."""
        results = self.trainer.train_best_model(
            self.X_train, self.y_train, self.X_test, self.y_test, 'classification'
        )
        
        assert 'best_model_name' in results
        assert 'best_score' in results
        assert 'models_tested' in results
        assert results['best_score'] > 0
        assert self.trainer.best_model is not None
    
    def test_predict(self):
        """Test prediction functionality."""
        # First train a model
        self.trainer.train_best_model(
            self.X_train, self.y_train, self.X_test, self.y_test, 'classification'
        )
        
        # Make predictions
        predictions = self.trainer.predict(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert all(pred in [0, 1] for pred in predictions)  # Binary classification
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        # Train and save model
        self.trainer.train_best_model(
            self.X_train, self.y_train, self.X_test, self.y_test, 'classification'
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            self.trainer.save_model(f.name)
            
            # Create new trainer and load
            new_trainer = Trainer()
            new_trainer.load_model(f.name)
            
            # Test that loaded model works
            predictions = new_trainer.predict(self.X_test)
            assert len(predictions) == len(self.X_test)
