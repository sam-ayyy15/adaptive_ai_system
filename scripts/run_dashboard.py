#!/usr/bin/env python3
"""
CLI entry point for launching the Streamlit dashboard.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    # Get the path to the dashboard module
    dashboard_path = Path(__file__).parent.parent / "src" / "dashboard.py"
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(dashboard_path), "--server.port", "8501"
    ]
    
    print("🚀 Starting Adaptive AI Dashboard...")
    print(f"📡 Dashboard will be available at: http://localhost:8501")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
