"""
Feedback ingestion and automated retraining system.
Handles user feedback and triggers model updates when threshold is reached.
"""
import pandas as pd
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json

from config import DATA_DIR, FEEDBACK_THRESHOLD, FEEDBACK_FILE
from trainer import Trainer
from data_handler import DataHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackSystem:
    """
    Automated feedback collection and retraining system.
    
    Example:
        >>> feedback = FeedbackSystem()
        >>> feedback.submit_feedback("new_data.csv", "target_column")
        >>> if feedback.should_retrain():
        ...     feedback.trigger_retraining()
    """
    
    def __init__(self):
        self.feedback_file = DATA_DIR / FEEDBACK_FILE
        self.trainer = Trainer()
        self.data_handler = DataHandler()
        self.feedback_counter = 0
        
        # Initialize feedback file if it doesn't exist
        if not self.feedback_file.exists():
            self._initialize_feedback_file()
        else:
            self._load_feedback_counter()
    
    def submit_feedback(self, csv_path: str, ground_truth_col: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Submit new feedback data for potential retraining.
        
        Args:
            csv_path: Path to CSV file with new data
            ground_truth_col: Name of the ground truth column
            metadata: Optional metadata about the feedback
            
        Returns:
            Feedback submission results
        """
        logger.info(f"Processing feedback from {csv_path}")
        
        try:
            # Load and validate new data
            new_data = pd.read_csv(csv_path)
            
            if ground_truth_col not in new_data.columns:
                raise ValueError(f"Ground truth column '{ground_truth_col}' not found")
            
            # Append to feedback file
            feedback_entry = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "data_shape": new_data.shape,
                "ground_truth_column": ground_truth_col,
                "source_file": csv_path,
                "metadata": metadata or {}
            }
            
            # Append data to feedback CSV
            self._append_feedback_data(new_data, feedback_entry)
            
            # Update counter
            self.feedback_counter += len(new_data)
            
            results = {
                "status": "success",
                "rows_added": len(new_data),
                "total_feedback_rows": self.feedback_counter,
                "threshold_reached": self.should_retrain(),
                "feedback_entry": feedback_entry
            }
            
            logger.info(f"Feedback submitted: {len(new_data)} rows, "
                       f"total: {self.feedback_counter}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return {"status": "error", "message": str(e)}
    
    def should_retrain(self) -> bool:
        """Check if retraining threshold has been reached."""
        return self.feedback_counter >= FEEDBACK_THRESHOLD
    
    def trigger_retraining(self, problem_type: str = "auto") -> Dict[str, Any]:
        """
        Trigger automated retraining with accumulated feedback.
        
        Args:
            problem_type: Type of ML problem or "auto" for detection
            
        Returns:
            Retraining results
        """
        if not self.should_retrain():
            return {
                "status": "skipped",
                "message": f"Threshold not reached. Current: {self.feedback_counter}, Required: {FEEDBACK_THRESHOLD}"
            }
        
        logger.info("Starting automated retraining with feedback data")
        
        try:
            # Load accumulated feedback data
            feedback_data = self._load_feedback_data()
            
            if feedback_data.empty:
                return {"status": "error", "message": "No feedback data available"}
            
            # Detect problem type if auto
            if problem_type == "auto":
                # This would need the target column name - simplified for demo
                problem_type = "classification"  # Default assumption
            
            # Prepare data for training
            target_col = self._get_target_column(feedback_data)
            
            if not target_col:
                return {"status": "error", "message": "Cannot determine target column"}
            
            # Split data
            X = feedback_data.drop(columns=[target_col])
            y = feedback_data[target_col]
            
            # Use data handler for proper splitting
            self.data_handler.data = feedback_data
            self.data_handler._analyze_dataset()
            X_train, X_test, y_train, y_test = self.data_handler.prepare_data(target_col)
            
            # Attempt warm start training if model exists
            results = {}
            try:
                # Try to load existing model
                self.trainer.load_model("best_model.pkl")
                results = self.trainer.warm_start_training(X_train, y_train, problem_type)
                training_type = "warm_start"
            except:
                # Full retraining
                results = self.trainer.train_best_model(X_train, y_train, X_test, y_test, problem_type)
                training_type = "full_retrain"
            
            # Save updated model
            self.trainer.save_model("best_model.pkl")
            
            # Reset feedback counter and archive data
            self._archive_feedback_data()
            self.feedback_counter = 0
            
            results.update({
                "status": "success",
                "training_type": training_type,
                "feedback_rows_used": len(feedback_data)
            })
            
            logger.info(f"Retraining completed successfully ({training_type})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_feedback_status(self) -> Dict[str, Any]:
        """Get current feedback system status."""
        return {
            "total_feedback_rows": self.feedback_counter,
            "threshold": FEEDBACK_THRESHOLD,
            "threshold_reached": self.should_retrain(),
            "progress_percentage": (self.feedback_counter / FEEDBACK_THRESHOLD) * 100,
            "feedback_file_exists": self.feedback_file.exists()
        }
    
    def _initialize_feedback_file(self) -> None:
        """Initialize empty feedback CSV file."""
        empty_df = pd.DataFrame()
        empty_df.to_csv(self.feedback_file, index=False)
        logger.info(f"Initialized feedback file: {self.feedback_file}")
    
    def _append_feedback_data(self, new_data: pd.DataFrame, 
                             feedback_entry: Dict[str, Any]) -> None:
        """Append new feedback data to the feedback file."""
        # Add metadata columns
        new_data = new_data.copy()
        new_data['feedback_timestamp'] = feedback_entry['timestamp']
        new_data['feedback_source'] = feedback_entry['source_file']
        
        # Append to existing file
        if self.feedback_file.exists() and self.feedback_file.stat().st_size > 0:
            new_data.to_csv(self.feedback_file, mode='a', header=False, index=False)
        else:
            new_data.to_csv(self.feedback_file, index=False)
    
    def _load_feedback_data(self) -> pd.DataFrame:
        """Load all accumulated feedback data."""
        if self.feedback_file.exists():
            return pd.read_csv(self.feedback_file)
        return pd.DataFrame()
    
    def _load_feedback_counter(self) -> None:
        """Load current feedback counter from file."""
        try:
            feedback_data = self._load_feedback_data()
            self.feedback_counter = len(feedback_data)
        except:
            self.feedback_counter = 0
    
    def _get_target_column(self, data: pd.DataFrame) -> Optional[str]:
        """Heuristically determine the target column."""
        # Remove metadata columns
        metadata_cols = ['feedback_timestamp', 'feedback_source']
        potential_targets = [col for col in data.columns if col not in metadata_cols]
        
        # Simple heuristic: assume last non-metadata column is target
        if potential_targets:
            return potential_targets[-1]
        return None
    
    def _archive_feedback_data(self) -> None:
        """Archive current feedback data after retraining."""
        if self.feedback_file.exists():
            archive_name = f"feedback_archive_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            archive_path = DATA_DIR / archive_name
            
            # Copy to archive
            feedback_data = self._load_feedback_data()
            feedback_data.to_csv(archive_path, index=False)
            
            # Clear original file
            self._initialize_feedback_file()
            
            logger.info(f"Feedback data archived to {archive_path}")


class AutoRetrainer:
    """
    Automated retraining scheduler and monitor.
    """
    
    def __init__(self):
        self.feedback_system = FeedbackSystem()
        self.is_monitoring = False
    
    def start_monitoring(self, check_interval: int = 3600) -> None:
        """Start monitoring for retraining conditions."""
        import threading
        import time
        
        def monitor():
            while self.is_monitoring:
                try:
                    if self.feedback_system.should_retrain():
                        logger.info("Automatic retraining triggered")
                        self.feedback_system.trigger_retraining()
                    
                    time.sleep(check_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
        
        self.is_monitoring = True
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
        logger.info(f"Started automatic retraining monitor (interval: {check_interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop automatic monitoring."""
        self.is_monitoring = False
        logger.info("Stopped automatic retraining monitor")
