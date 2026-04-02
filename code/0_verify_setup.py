"""
Step 0: Verify Setup
Verify environment, data, and pre-trained model artifacts.
Run this first after cloning the repository.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("VERIFY SETUP: Verifying environment")
    print("=" * 60)

    # Python version
    print(f"\nPython: {sys.version}")

    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("ERROR: PyTorch not installed. Run: uv sync")
        sys.exit(1)

    # Key dependencies
    for pkg in ["pandas", "numpy", "sklearn", "dash", "plotly"]:
        try:
            mod = __import__(pkg)
            print(f"{pkg}: {mod.__version__}")
        except ImportError:
            print(f"WARNING: {pkg} not installed")

    # Dataset
    data_path = PROJECT_ROOT / "data" / "nyc_taxi.csv"
    if data_path.exists():
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"\nDataset: {data_path}")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    else:
        print(f"\nWARNING: Dataset not found at {data_path}")

    # Pre-trained model artifacts
    models_dir = PROJECT_ROOT / "models"
    artifacts = ["lstm_model.pt", "scaler.pkl", "scorer.pkl"]
    print(f"\nModel artifacts ({models_dir}):")
    for name in artifacts:
        path = models_dir / name
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  {name}: {size_kb:.1f} KB")
        else:
            print(f"  {name}: NOT FOUND (run code/3_train_model.py to train)")

    print("\n" + "=" * 60)
    print("Verify Setup complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
