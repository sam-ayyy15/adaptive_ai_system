#!/usr/bin/env python3
"""
CLI script for batch predictions on new CSV files.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trainer import Trainer
from data_handler import DataHandler


def main():
    """Main batch prediction function."""
    parser = argparse.ArgumentParser(description="Batch prediction on CSV files")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument("--model", default="best_model.pkl", help="Model file to use")
    
    args = parser.parse_args()
    
    try:
        print(f"📂 Loading model from {args.model}...")
        trainer = Trainer()
        trainer.load_model(args.model)
        
        print(f"📊 Loading data from {args.input_csv}...")
        data = pd.read_csv(args.input_csv)
        print(f"✅ Loaded {len(data)} rows")
        
        print("🔮 Making predictions...")
        predictions = trainer.predict(data)
        
        # Add predictions to dataframe
        result_df = data.copy()
        result_df['Prediction'] = predictions
        
        # Save results
        result_df.to_csv(args.output_csv, index=False)
        print(f"💾 Predictions saved to {args.output_csv}")
        
        # Print summary
        if pd.api.types.is_numeric_dtype(result_df['Prediction']):
            print(f"📈 Prediction statistics:")
            print(f"   Mean: {predictions.mean():.4f}")
            print(f"   Std:  {predictions.std():.4f}")
            print(f"   Min:  {predictions.min():.4f}")
            print(f"   Max:  {predictions.max():.4f}")
        else:
            print(f"📊 Prediction distribution:")
            print(pd.Series(predictions).value_counts().head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
