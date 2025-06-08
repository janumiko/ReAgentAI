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
        self.experiment_name = experiment_name
        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        self.active_run = None
        self.mlflow_enabled = bool(self.tracking_uri)

        if not self.mlflow_enabled:
            logger.info("MLflow tracking is disabled - MLFLOW_TRACKING_URI is not set.")
            return

        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            # Create or get the experiment
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if not self.experiment:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id

            self.client = MlflowClient()
        except Exception as e:
            logger.warning(f"MLflow experiment setup failed: {e}")
            self.mlflow_enabled = False

    def start_run(self, run_name: str | None = None) -> str | None:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run

        Returns:
            The run ID of the created run or None if MLflow is disabled
        """
        if not self.mlflow_enabled:
            logger.debug("MLflow tracking disabled. Not starting run.")
            return None

        try:
            self.active_run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
            return self.active_run.info.run_id
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            self.mlflow_enabled = False
            return None

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.mlflow_enabled or not self.active_run:
            return

        try:
            mlflow.end_run()
            self.active_run = None
        except Exception as e:
            logger.warning(f"Error ending MLflow run: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to the current run."""
        if not self.mlflow_enabled or not self.active_run:
            return

        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")

    def log_metrics(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log metrics to the current run."""
        if not self.mlflow_enabled or not self.active_run:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, local_path: str) -> None:
        """Log an artifact to the current run."""
        if not self.mlflow_enabled or not self.active_run:
            return

        try:
            mlflow.log_artifact(local_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on the current run."""
        if not self.mlflow_enabled or not self.active_run:
            return

        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.warning(f"Failed to set tags in MLflow: {e}")
