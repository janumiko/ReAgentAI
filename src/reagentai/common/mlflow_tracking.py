import logging
import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Handles MLflow experiment tracking for ReagentAI.
    """

    def __init__(self, experiment_name: str = "reagentai_experiments"):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: The name of the MLflow experiment to use
        """
        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlflow")
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get the experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if not self.experiment:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            self.experiment_id = None

        self.client = MlflowClient()
        self.active_run = None

    def start_run(self, run_name: str | None = None) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run

        Returns:
            The run ID of the created run
        """
        if self.experiment_id is None:
            logger.warning("MLflow experiment not properly initialized. Tracking disabled.")
            return None

        self.active_run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        return self.active_run.info.run_id

    def end_run(self):
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params: dict[str, Any]):
        """Log parameters to the current run."""
        if self.active_run:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float | int], step: int | None = None):
        """Log metrics to the current run."""
        if self.active_run:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str):
        """Log an artifact to the current run."""
        if self.active_run:
            mlflow.log_artifact(local_path)

    def set_tags(self, tags: dict[str, str]):
        """Set tags on the current run."""
        if self.active_run:
            mlflow.set_tags(tags)
