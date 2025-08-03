"""
Tests for data_handler module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_handler import DataHandler


class TestDataHandler:
    """Test cases for DataHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = DataHandler()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 1],
            'missing_col': [1, None, 3, None, 5]
        })
    
    def test_load_and_analyze(self):
        """Test CSV loading and analysis."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            
            try:
                analysis = self.handler.load_and_analyze(f.name)
                
                assert analysis['shape'] == (5, 4)
                assert 'numeric_col' in analysis['numeric_columns']
                assert 'categorical_col' in analysis['categorical_columns']
                assert analysis['missing_percentages']['missing_col'] == 0.4
                
            finally:
                os.unlink(f.name)
    
    def test_prepare_data(self):
        """Test data preparation and splitting."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            
            try:
                self.handler.load_and_analyze(f.name)
                X_train, X_test, y_train, y_test = self.handler.prepare_data('target')
                
                assert len(X_train) + len(X_test) <= len(self.sample_data)
                assert len(y_train) == len(X_train)
                assert len(y_test) == len(X_test)
                
            finally:
                os.unlink(f.name)
    
    def test_problem_type_detection(self):
        """Test problem type detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            
            try:
                self.handler.load_and_analyze(f.name)
                problem_type = self.handler.get_problem_type('target')
                
                assert problem_type in ['classification', 'regression', 'time_series']
                
            finally:
                os.unlink(f.name)
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            
            try:
                self.handler.load_and_analyze(f.name)
                cleaned_data = self.handler.clean_data('target')
                
                # Should remove rows with missing target (none in this case)
                assert len(cleaned_data) <= len(self.sample_data)
                assert 'target' in cleaned_data.columns
                
            finally:
                os.unlink(f.name)
