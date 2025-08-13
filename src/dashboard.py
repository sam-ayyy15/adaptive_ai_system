"""
Streamlit dashboard for interactive model training, monitoring, and predictions.
Provides real-time interface for the adaptive AI system.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
from typing import Dict, Any, Optional
import logging

from config import DATA_DIR, MODELS_DIR, REFRESH_INTERVAL
from data_handler import DataHandler
from trainer import Trainer
from feedback import FeedbackSystem
from explainability import ExplainabilityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveDashboard:
    """
    Streamlit-based interactive dashboard for the adaptive AI system.
    """
    
    def __init__(self):
        self.data_handler = DataHandler()
        self.trainer = Trainer()
        self.feedback_system = FeedbackSystem()
        self.explainer = ExplainabilityEngine()
        
        # Initialize session state
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'training_results' not in st.session_state:
            st.session_state.training_results = None
    
    def run(self):
        """Main dashboard application."""
        st.set_page_config(
            page_title="Adaptive AI System",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title(" Intelligent Regression Analytics: A Machine Learning Predctive Model")
        st.markdown("*Universal AutoML Platform for Tabular Data*")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate",
            ["Overview", "Data Upload", "Model Training", "Predictions", 
             "Feature Importance", "Feedback", "System Status"]
        )
        
        # Route to appropriate page
        if page == "Overview":
            self.show_overview()
        elif page == "Data Upload":
            self.show_data_upload()
        elif page == "Model Training":
            self.show_model_training()
        elif page == "Predictions":
            self.show_predictions()
        elif page == "Feature Importance":
            self.show_feature_importance()
        elif page == "Feedback":
            self.show_feedback()
        elif page == "System Status":
            self.show_system_status()
    
    def show_overview(self):
        """Display system overview and architecture."""
        st.header("System Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Available", "7", delta="XGB, RF, GB, etc.")
        
        with col2:
            feedback_status = self.feedback_system.get_feedback_status()
            st.metric(
                "Feedback Rows", 
                feedback_status['total_feedback_rows'],
                delta=f"{feedback_status['progress_percentage']:.1f}% to retrain"
            )
        
        with col3:
            st.metric("Auto Features", "✓", delta="Type detection, encoding")
        
        st.markdown("---")
        
        # System Architecture
        st.subheader("System Architecture")
        
        architecture_ascii = """
        ```
        📁 Data Upload → 🔍 Auto Analysis → 🧠 Model Selection
               ↓                ↓               ↓
        🛠️ Feature Eng. → ⚡ Training → 📊 Evaluation
               ↓                ↓               ↓
        💡 Explainability ← 🔄 Feedback ← 📈 Monitoring
        ```
        """
        st.markdown(architecture_ascii)
        
        # Recent Activity
        st.subheader("Recent Activity")
        
        # Mock recent activity for demo
        activity_data = pd.DataFrame({
            'Timestamp': pd.date_range('2024-01-01', periods=5, freq='H'),
            'Activity': ['Data uploaded', 'Model trained', 'Predictions made', 
                        'Feedback received', 'Model retrained'],
            'Status': ['✅', '✅', '✅', '⏳', '✅']
        })
        
        st.dataframe(activity_data, use_container_width=True)
    
    def show_data_upload(self):
        """Data upload and analysis interface."""
        st.header("📁 Data Upload & Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your tabular dataset for analysis and training"
        )
        
        if uploaded_file is not None:
            try:
                # Load and analyze data
                data = pd.read_csv(uploaded_file)
                st.session_state.current_data = data
                
                # Display basic info
                st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                
                # Data analysis
                st.subheader("Automated Data Analysis")
                
                # Save temp file for analysis
                temp_path = DATA_DIR / "temp_upload.csv"
                data.to_csv(temp_path, index=False)
                
                # Analyze with data handler
                analysis = self.data_handler.load_and_analyze(str(temp_path))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Column Types**")
                    type_counts = pd.Series(analysis['column_types']).value_counts()
                    fig = px.pie(values=type_counts.values, names=type_counts.index,
                               title="Feature Types Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Missing Values**")
                    missing_data = pd.DataFrame({
                        'Column': list(analysis['missing_percentages'].keys()),
                        'Missing %': [v*100 for v in analysis['missing_percentages'].values()]
                    })
                    missing_data = missing_data[missing_data['Missing %'] > 0]
                    
                    if not missing_data.empty:
                        fig = px.bar(missing_data, x='Column', y='Missing %',
                                   title="Missing Values by Column")
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("No missing values detected!")
                
                # Column details
                st.subheader("Column Details")
                
                details_df = pd.DataFrame({
                    'Column': analysis['columns'],
                    'Type': [analysis['column_types'][col] for col in analysis['columns']],
                    'Unique Values': [analysis['cardinality'][col] for col in analysis['columns']],
                    'Missing %': [f"{analysis['missing_percentages'][col]*100:.1f}%" 
                                for col in analysis['columns']]
                })
                
                st.dataframe(details_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    def show_model_training(self):
        """Model training interface."""
        st.header("🧠 Model Training")
        
        if st.session_state.current_data is None:
            st.warning("Please upload data first in the 'Data Upload' tab.")
            return
        
        data = st.session_state.current_data
        
        # Target selection
        st.subheader("Training Configuration")
        
        target_column = st.selectbox(
            "Select Target Column",
            data.columns.tolist(),
            help="Choose the column you want to predict"
        )
        
        problem_type = st.selectbox(
            "Problem Type",
            ["auto", "classification", "regression", "time_series"],
            help="Select problem type or let the system detect automatically"
        )
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Prepare data
                    progress_bar = st.progress(0, text="Preparing data...")
                    
                    # Save data and analyze
                    temp_path = DATA_DIR / "training_data.csv"
                    data.to_csv(temp_path, index=False)
                    
                    self.data_handler.load_and_analyze(str(temp_path))
                    progress_bar.progress(20, text="Data analysis complete...")
                    
                    # Determine problem type
                    if problem_type == "auto":
                        detected_type = self.data_handler.get_problem_type(target_column)
                        st.info(f"Detected problem type: {detected_type}")
                        problem_type = detected_type
                    
                    # Split data
                    X_train, X_test, y_train, y_test = self.data_handler.prepare_data(target_column)
                    # Persist the selected target for later prediction UIs
                    st.session_state.target_column = target_column
                    progress_bar.progress(40, text="Data split complete...")
                    
                    # Train models
                    progress_bar.progress(60, text="Training models...")
                    results = self.trainer.train_best_model(
                        X_train, y_train, X_test, y_test, problem_type
                    )
                    
                    progress_bar.progress(80, text="Saving model...")
                    
                    # Save model
                    self.trainer.save_model("best_model.pkl")
                    
                    progress_bar.progress(100, text="Training complete!")
                    
                    # Store results
                    st.session_state.training_results = results
                    st.session_state.model_trained = True
                    
                    st.success("🎉 Training completed successfully!")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    logger.error(f"Training error: {e}")
        
        # Display results if available
        if st.session_state.training_results:
            self.display_training_results(st.session_state.training_results)
    
    def display_training_results(self, results: Dict[str, Any]):
        """Display training results."""
        st.subheader("Training Results")
        
        # Best model info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Model", results['best_model_name'])
        
        with col2:
            st.metric("Best Score", f"{results['best_score']:.4f}")
        
        with col3:
            st.metric("Models Tested", len(results['models_tested']))
        
        # Model comparison
        if len(results['test_scores']) > 1:
            st.subheader("Model Comparison")
            
            comparison_df = pd.DataFrame({
                'Model': list(results['test_scores'].keys()),
                'Test Score': list(results['test_scores'].values()),
                'CV Score': [results['cross_val_scores'].get(model, 0) 
                           for model in results['test_scores'].keys()]
            })
            
            fig = px.bar(comparison_df, x='Model', y=['Test Score', 'CV Score'],
                        title="Model Performance Comparison", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if results.get('feature_importance'):
            st.subheader("Top Feature Importance")
            
            importance_df = pd.DataFrame([
                {'Feature': k, 'Importance': v} 
                for k, v in list(results['feature_importance'].items())[:10]
            ])
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h', title="Top 10 Most Important Features")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_predictions(self):
        """Prediction interface."""
        st.header("📈 Make Predictions")
        
        if not st.session_state.model_trained:
            st.warning("Please train a model first in the 'Model Training' tab.")
            return
        
        # Load model once and cache across reruns for speed
        @st.cache_resource(show_spinner=False)
        def _load_trainer_cached() -> Trainer:
            trainer_inst = Trainer()
            trainer_inst.load_model("best_model.pkl")
            return trainer_inst

        try:
            self.trainer = _load_trainer_cached()
        except Exception as e:
            st.error(f"Could not load trained model: {e}")
            return
        
        # Prediction options
        prediction_type = st.radio(
            "Prediction Type",
            ["Single Prediction", "Batch Prediction"]
        )
        
        if prediction_type == "Single Prediction":
            self.show_single_prediction()
        else:
            self.show_batch_prediction()
    
    def show_single_prediction(self):
        """Interface for single instance prediction."""
        st.subheader("Single Instance Prediction")
        
        if st.session_state.current_data is None:
            st.warning("No data available for reference.")
            return
        
        data = st.session_state.current_data
        target_col = st.session_state.get("target_column")
        
        # Create input form
        st.markdown("Enter values for prediction:")
        
        # Get feature columns (excluding target)
        all_columns = [c for c in data.columns.tolist() if c != target_col]
        
        # Create input form with all columns
        input_data = {}
        
        # Create multiple columns for better layout
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, column in enumerate(all_columns):
            col_idx = i % num_cols
            with cols[col_idx]:
                if data[column].dtype in ['int64', 'float64']:
                    input_data[column] = st.number_input(
                        f"{column}",
                        value=float(data[column].mean()),
                        format="%.6f"
                    )
                else:
                    unique_values = data[column].unique()[:10]  # Limit options
                    input_data[column] = st.selectbox(
                        f"{column}",
                        unique_values
                    )
        
        if st.button("Predict"):
            try:
                # Create prediction DataFrame
                pred_df = pd.DataFrame([input_data])

                # Prefer probabilities when available (classification)
                showed_probability = False
                try:
                    probabilities = self.trainer.predict_proba(pred_df)
                    if probabilities is not None and len(probabilities.shape) == 2:
                        if probabilities.shape[1] == 2:
                            st.success(f"Probability: {float(probabilities[0][1]):.6f}")
                            showed_probability = True
                        else:
                            st.subheader("Prediction Probabilities")
                            prob_df = pd.DataFrame({
                                'Class': range(len(probabilities[0])),
                                'Probability': probabilities[0]
                            })
                            fig = px.bar(prob_df, x='Class', y='Probability', title="Class Probabilities")
                            st.plotly_chart(fig)
                            showed_probability = True
                except Exception:
                    pass

                if not showed_probability:
                    # Fall back to raw prediction (e.g., regression models)
                    prediction = self.trainer.predict(pred_df)
                    try:
                        val = float(prediction[0])
                    except Exception:
                        val = float(prediction)

                    # If outside [0,1], map to a probability via logistic
                    if val < 0.0 or val > 1.0:
                        import math
                        prob = 1.0 / (1.0 + math.exp(-val))
                        st.success(f"Probability : {prob:.6f}")
                    else:
                        st.success(f"Probability: {val:.6f}")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    def show_batch_prediction(self):
        """Interface for batch predictions."""
        st.subheader("Batch Prediction")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch prediction",
            type="csv"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                pred_data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(pred_data)} rows for prediction")
                
                # Preview
                st.subheader("Data Preview")
                st.dataframe(pred_data.head())
                
                if st.button("Generate Predictions"):
                    with st.spinner("Making predictions..."):
                        result_df = pred_data.copy()

                        # Try probabilities first
                        added_proba = False
                        try:
                            probas = self.trainer.predict_proba(pred_data)
                            if probas is not None and len(probas.shape) == 2:
                                if probas.shape[1] == 2:
                                    result_df['Probability'] = probas[:, 1]
                                    added_proba = True
                                else:
                                    for i in range(probas.shape[1]):
                                        result_df[f'Prob_class_{i}'] = probas[:, i]
                                    added_proba = True
                        except Exception:
                            pass

                        if not added_proba:
                            # Fall back to raw predictions; map to probability if needed
                            predictions = self.trainer.predict(pred_data)
                            try:
                                import numpy as _np
                                preds_array = _np.array(predictions, dtype=float)
                                needs_map = (preds_array < 0).any() or (preds_array > 1).any()
                                if needs_map:
                                    preds_array = 1.0 / (1.0 + _np.exp(-preds_array))
                                result_df['Probability'] = preds_array
                            except Exception:
                                result_df['Prediction'] = predictions
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(result_df)
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="Download Predictions",
                            data=csv_buffer.getvalue(),
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error in batch prediction: {str(e)}")
    
    def show_feature_importance(self):
        """Feature importance and explainability interface."""
        st.header("💡 Feature Importance & Explainability")
        
        if not st.session_state.model_trained:
            st.warning("Please train a model first.")
            return
        
        if not st.session_state.training_results:
            st.warning("No training results available.")
            return
        
        results = st.session_state.training_results
        
        # Global feature importance
        if results.get('feature_importance'):
            st.subheader("Global Feature Importance")
            
            importance_df = pd.DataFrame([
                {'Feature': k, 'Importance': v} 
                for k, v in results['feature_importance'].items()
            ])
            
            # Interactive plot
            fig = px.bar(importance_df.head(15), x='Importance', y='Feature',
                        orientation='h', title="Top 15 Most Important Features")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("Feature Importance Details")
            st.dataframe(importance_df, use_container_width=True)
        
        # SHAP explanations (placeholder)
        st.subheader("SHAP Explanations")
        st.info("SHAP explanations would be generated here for deeper insights into model predictions.")
        
        # Model insights
        st.subheader("Model Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Characteristics**")
            st.write(f"- **Algorithm**: {results['best_model_name']}")
            st.write(f"- **Performance**: {results['best_score']:.4f}")
            st.write(f"- **Features Used**: {len(results.get('feature_importance', {}))}")
        
        with col2:
            st.markdown("**Dataset Insights**")
            if 'analysis' in results:
                profile = results['analysis']['dataset_profile']
                st.write(f"- **Samples**: {profile['n_samples']:,}")
                st.write(f"- **Features**: {profile['n_features']}")
                st.write(f"- **Problem Type**: {profile['problem_type']}")
    
    def show_feedback(self):
        """Feedback submission interface."""
        st.header("🔄 Feedback & Continuous Learning")
        
        # Feedback status
        status = self.feedback_system.get_feedback_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Feedback Rows", status['total_feedback_rows'])
        
        with col2:
            st.metric("Threshold", status['threshold'])
        
        with col3:
            progress = status['progress_percentage']
            st.metric("Progress", f"{progress:.1f}%")
        
        # Progress bar
        st.progress(min(progress/100, 1.0))
        
        if status['threshold_reached']:
            st.success("🎉 Retraining threshold reached! Model can be updated.")
            
            if st.button("Trigger Retraining", type="primary"):
                with st.spinner("Retraining model with feedback..."):
                    try:
                        retrain_results = self.feedback_system.trigger_retraining()
                        if retrain_results['status'] == 'success':
                            st.success("Model retrained successfully!")
                            st.json(retrain_results)
                        else:
                            st.error(f"Retraining failed: {retrain_results.get('message', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error during retraining: {str(e)}")
        
        # Feedback submission
        st.subheader("Submit New Feedback Data")
        
        feedback_file = st.file_uploader(
            "Upload CSV with ground truth",
            type="csv",
            help="Upload a CSV file containing new data with actual outcomes"
        )
        
        if feedback_file is not None:
            try:
                feedback_data = pd.read_csv(feedback_file)
                st.success(f"Loaded feedback data: {feedback_data.shape}")
                
                # Preview
                st.dataframe(feedback_data.head())
                
                # Ground truth column selection
                ground_truth_col = st.selectbox(
                    "Select Ground Truth Column",
                    feedback_data.columns.tolist()
                )
                
                if st.button("Submit Feedback"):
                    # Save temporary file
                    temp_path = DATA_DIR / "temp_feedback.csv"
                    feedback_data.to_csv(temp_path, index=False)
                    
                    # Submit feedback
                    result = self.feedback_system.submit_feedback(
                        str(temp_path), ground_truth_col
                    )
                    
                    if result['status'] == 'success':
                        st.success(f"Feedback submitted! Added {result['rows_added']} rows.")
                        
                        # Refresh page to update metrics
                        st.rerun()
                    else:
                        st.error(f"Feedback submission failed: {result.get('message', 'Unknown error')}")
                        
            except Exception as e:
                st.error(f"Error processing feedback: {str(e)}")
    
    def show_system_status(self):
        """System status and monitoring interface."""
        st.header("🔧 System Status")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
        
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        
        # System health
        st.subheader("System Health")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Handler", "✅ Online")
        
        with col2:
            model_status = "✅ Trained" if st.session_state.model_trained else "⚠️ Not Trained"
            st.metric("Model Status", model_status)
        
        with col3:
            st.metric("Feedback System", "✅ Active")
        
        with col4:
            st.metric("Dashboard", "✅ Running")
        
        # Resource usage (mock data)
        st.subheader("Resource Usage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU usage chart
            cpu_data = pd.DataFrame({
                'Time': pd.date_range('now', periods=20, freq='1min'),
                'CPU %': np.random.randint(20, 80, 20)
            })
            
            fig = px.line(cpu_data, x='Time', y='CPU %', title="CPU Usage")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Memory usage chart
            memory_data = pd.DataFrame({
                'Time': pd.date_range('now', periods=20, freq='1min'),
                'Memory %': np.random.randint(30, 70, 20)
            })
            
            fig = px.line(memory_data, x='Time', y='Memory %', title="Memory Usage")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent logs
        st.subheader("Recent Activity Logs")
        
        logs = [
            {"Timestamp": "2024-01-15 10:30:00", "Level": "INFO", "Message": "Model training completed"},
            {"Timestamp": "2024-01-15 10:25:00", "Level": "INFO", "Message": "Data uploaded and analyzed"},
            {"Timestamp": "2024-01-15 10:20:00", "Level": "INFO", "Message": "Feedback data received"},
            {"Timestamp": "2024-01-15 10:15:00", "Level": "INFO", "Message": "Dashboard started"},
        ]
        
        log_df = pd.DataFrame(logs)
        st.dataframe(log_df, use_container_width=True)


def run_dashboard():
    """Entry point for running the dashboard."""
    dashboard = AdaptiveDashboard()
    dashboard.run()


if __name__ == "__main__":
    run_dashboard()
