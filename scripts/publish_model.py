#!/usr/bin/env python3
"""
Optional script for publishing models to a registry or cloud service.
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import MODELS_DIR


def main():
    """Main model publishing function."""
    parser = argparse.ArgumentParser(description="Publish model to registry")
    parser.add_argument("--model", default="best_model.pkl", help="Model file to publish")
    parser.add_argument("--registry", default="local", help="Registry type (local/s3/gcs)")
    parser.add_argument("--name", help="Model name in registry")
    parser.add_argument("--version", help="Model version")
    
    args = parser.parse_args()
    
    model_path = MODELS_DIR / args.model
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    # Generate model metadata
    metadata = {
        "model_file": args.model,
        "registry": args.registry,
        "name": args.name or args.model.replace('.pkl', ''),
        "version": args.version or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "published_at": datetime.now().isoformat(),
        "file_size": model_path.stat().st_size
    }
    
    print(f"📦 Publishing model: {metadata['name']} v{metadata['version']}")
    
    if args.registry == "local":
        # Local registry (just copy with metadata)
        registry_dir = MODELS_DIR / "registry"
        registry_dir.mkdir(exist_ok=True)
        
        # Copy model file
        published_path = registry_dir / f"{metadata['name']}_v{metadata['version']}.pkl"
        import shutil
        shutil.copy2(model_path, published_path)
        
        # Save metadata
        metadata_path = registry_dir / f"{metadata['name']}_v{metadata['version']}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model published to local registry:")
        print(f"   Model: {published_path}")
        print(f"   Metadata: {metadata_path}")
        
    elif args.registry == "s3":
        print("🔄 S3 registry not implemented in this demo")
        print("   Would upload to: s3://your-bucket/models/")
        
    elif args.registry == "gcs":
        print("🔄 GCS registry not implemented in this demo") 
        print("   Would upload to: gs://your-bucket/models/")
        
    else:
        print(f"❌ Unknown registry type: {args.registry}")
        sys.exit(1)


if __name__ == "__main__":
    main()
