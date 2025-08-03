"""
Data handling module for automatic type detection, cleaning, and splitting.
Provides robust CSV processing with intelligent preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, TEST_SIZE, CATEGORICAL_THRESHOLD, MISSING_VALUE_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """
    Handles CSV data loading, type detection, cleaning, and preprocessing.
    
    Example:
        >>> handler = DataHandler()
        >>> data_info = handler.load_and_analyze("data.csv")
        >>> X_train, X_test, y_train, y_test = handler.prepare_data("target_col")
    """
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.data_info: Dict[str, Any] = {}
        
    def load_and_analyze(self, csv_path: str) -> Dict[str, Any]:
        """
        Load CSV file and perform comprehensive data analysis.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Dictionary containing data analysis results
        """
        try:
            # Load data with automatic parsing
            self.data = pd.read_csv(csv_path, parse_dates=True, infer_datetime_format=True)
            logger.info(f"Loaded dataset with shape: {self.data.shape}")
            
            # Perform analysis
            self.data_info = self._analyze_dataset()
            return self.data_info
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def _analyze_dataset(self) -> Dict[str, Any]:
        """Comprehensive dataset analysis and profiling."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        analysis = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "missing_percentages": (self.data.isnull().sum() / len(self.data)).to_dict(),
            "column_types": {},
            "cardinality": {},
            "has_datetime": False,
            "datetime_columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "high_cardinality_columns": []
        }
        
        # Analyze each column
        for col in self.data.columns:
            col_data = self.data[col].dropna()
            
            # Cardinality
            unique_count = col_data.nunique()
            analysis["cardinality"][col] = unique_count
            
            # Type detection
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                analysis["column_types"][col] = "datetime"
                analysis["datetime_columns"].append(col)
                analysis["has_datetime"] = True
            elif pd.api.types.is_numeric_dtype(self.data[col]):
                analysis["column_types"][col] = "numeric"
                analysis["numeric_columns"].append(col)
            elif unique_count <= CATEGORICAL_THRESHOLD:
                analysis["column_types"][col] = "categorical"
                analysis["categorical_columns"].append(col)
            else:
                analysis["column_types"][col] = "high_cardinality_categorical"
                analysis["high_cardinality_columns"].append(col)
        
        # Overall statistics
        analysis["total_missing"] = self.data.isnull().sum().sum()
        analysis["missing_percentage"] = analysis["total_missing"] / (self.data.shape[0] * self.data.shape[1])
        
        logger.info(f"Analysis complete: {len(analysis['numeric_columns'])} numeric, "
                   f"{len(analysis['categorical_columns'])} categorical, "
                   f"{len(analysis['datetime_columns'])} datetime columns")
        
        return analysis
    
    def clean_data(self, target_column: str) -> pd.DataFrame:
        """
        Clean dataset by handling missing values and outliers.
        
        Args:
            target_column: Name of target column
            
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        df_clean = self.data.copy()
        
        # Remove columns with excessive missing values
        high_missing_cols = [
            col for col, pct in self.data_info["missing_percentages"].items()
            if pct > MISSING_VALUE_THRESHOLD and col != target_column
        ]
        
        if high_missing_cols:
            logger.info(f"Dropping columns with >50% missing: {high_missing_cols}")
            df_clean = df_clean.drop(columns=high_missing_cols)
        
        # Remove rows where target is missing
        if target_column in df_clean.columns:
            df_clean = df_clean.dropna(subset=[target_column])
            logger.info(f"Removed {len(self.data) - len(df_clean)} rows with missing target")
        
        return df_clean
    
    def prepare_data(self, target_column: str, test_size: float = TEST_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting into features and target.
        
        Args:
            target_column: Name of target column
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Clean data
        df_clean = self.clean_data(target_column)
        
        # Separate features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED,
            stratify=y if self._is_classification_target(y) else None
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _is_classification_target(self, y: pd.Series) -> bool:
        """Determine if target is classification or regression."""
        if pd.api.types.is_numeric_dtype(y):
            unique_count = y.nunique()
            # If we have very few unique values (like binary classification), it's classification
            if unique_count <= 10:
                return True
            # If unique ratio is very small, it's likely classification
            unique_ratio = unique_count / len(y)
            return unique_ratio < 0.1
        return True
    
    def get_problem_type(self, target_column: str) -> str:
        """
        Determine problem type (classification/regression/time_series).
        
        Args:
            target_column: Name of target column
            
        Returns:
            Problem type string
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        y = self.data[target_column]
        
        # Check for time series
        if self.data_info["has_datetime"]:
            return "time_series"
        
        # Classification vs regression
        return "classification" if self._is_classification_target(y) else "regression"
    
    def export_cleaned_data(self, output_path: str, target_column: str) -> None:
        """Export cleaned dataset to CSV."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        df_clean = self.clean_data(target_column)
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Cleaned data exported to {output_path}")
