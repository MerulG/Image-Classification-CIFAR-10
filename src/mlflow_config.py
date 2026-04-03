"""
Centralises MLflow tracking configuration so experiment name and storage
location are defined in one place.
"""

from pathlib import Path

import mlflow

EXPERIMENT_NAME = "cifar10-cnn"
# Absolute path so the same mlruns/ directory is used regardless of where
# the script is invoked from (e.g. src/ vs project root).
TRACKING_URI = str(Path(__file__).parent.parent / "mlruns")


def setup_mlflow():
    """Configure MLflow tracking URI and experiment, returning the active experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment: '{EXPERIMENT_NAME}' | Tracking URI: '{TRACKING_URI}'")
    return experiment
