import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.reagentai.common.mlflow_tracking import MLflowTracker


class TestReAgentAI:
    """End-to-end tests for ReAgentAI application."""

    @pytest.fixture(scope="class")
    def mlflow_tracker(self):
        """Create a MLflowTracker for testing."""
        # Use a temporary directory as a valid file URI
        temp_dir = tempfile.mkdtemp()
        tracking_uri = f"file://{temp_dir}"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        tracker = MLflowTracker(experiment_name="test_experiment")
        yield tracker
        # Clean up
        tracker.end_run()
        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]

    def test_mlflow_tracker_initialization(self):
        """Test MLflowTracker initialization with different configurations."""
        # Create a temporary directory for a valid mlflow file URI
        temp_dir = tempfile.mkdtemp()
        valid_uri = f"file://{temp_dir}"

        # Test with valid tracking URI
        os.environ["MLFLOW_TRACKING_URI"] = valid_uri
        with patch("mlflow.set_tracking_uri") as mock_set_uri:
            with patch("mlflow.get_experiment_by_name") as mock_get_exp:
                # Set up the mock to return a valid experiment
                mock_exp = MagicMock()
                mock_exp.experiment_id = "test-exp-id"
                mock_get_exp.return_value = mock_exp

                tracker = MLflowTracker()
                assert tracker.mlflow_enabled is True
                assert tracker.tracking_uri == valid_uri
                mock_set_uri.assert_called_once_with(valid_uri)

        # Test without tracking URI
        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]
        tracker = MLflowTracker()
        assert tracker.mlflow_enabled is False
        assert tracker.tracking_uri == ""

    def test_mlflow_tracker_lifecycle(self):
        """Test the full lifecycle of MLflowTracker with patching at the right level."""
        # Create a temporary directory for a valid mlflow file URI
        temp_dir = tempfile.mkdtemp()
        valid_uri = f"file://{temp_dir}"
        os.environ["MLFLOW_TRACKING_URI"] = valid_uri

        # Define a class to simulate the MLflow implementation
        class MockMLflow:
            @staticmethod
            def set_tracking_uri(uri):
                pass

            @staticmethod
            def get_experiment_by_name(name):
                mock_exp = MagicMock()
                mock_exp.experiment_id = "test-exp-id"
                return mock_exp

            @staticmethod
            def create_experiment(name):
                return "new-exp-id"

            @staticmethod
            def start_run(experiment_id=None, run_name=None):
                mock_run = MagicMock()
                mock_run.info.run_id = "test-run-id"
                return mock_run

            @staticmethod
            def end_run():
                pass

            @staticmethod
            def log_params(params):
                pass

            @staticmethod
            def log_metrics(metrics, step=None):
                pass

            @staticmethod
            def log_artifact(local_path):
                pass

            @staticmethod
            def set_tags(tags):
                pass

        # Patch the entire mlflow module with our MockMLflow
        with patch.object(
            sys.modules["src.reagentai.common.mlflow_tracking"], "mlflow", MockMLflow
        ):
            # Now initialize the tracker with our mocked MLflow
            tracker = MLflowTracker(experiment_name="test_experiment")
            assert tracker.mlflow_enabled is True

            # Spy on the methods to verify they're called
            with patch.object(
                MockMLflow, "start_run", wraps=MockMLflow.start_run
            ) as mock_start_run:
                run_id = tracker.start_run(run_name="test_run")
                assert run_id == "test-run-id"
                mock_start_run.assert_called_once_with(
                    experiment_id="test-exp-id", run_name="test_run"
                )

            # Test log_params
            with patch.object(
                MockMLflow, "log_params", wraps=MockMLflow.log_params
            ) as mock_log_params:
                params = {"param1": "value1", "param2": 42}
                tracker.log_params(params)
                mock_log_params.assert_called_once_with(params)

            # Test log_metrics
            with patch.object(
                MockMLflow, "log_metrics", wraps=MockMLflow.log_metrics
            ) as mock_log_metrics:
                metrics = {"metric1": 0.95, "metric2": 0.87}
                tracker.log_metrics(metrics)
                mock_log_metrics.assert_called_once_with(metrics, step=None)

            # Test log_artifact
            with patch.object(
                MockMLflow, "log_artifact", wraps=MockMLflow.log_artifact
            ) as mock_log_artifact:
                test_file = "test_artifact.txt"
                tracker.log_artifact(test_file)
                mock_log_artifact.assert_called_once_with(test_file)

            # Test set_tags
            with patch.object(MockMLflow, "set_tags", wraps=MockMLflow.set_tags) as mock_set_tags:
                tags = {"tag1": "value1", "tag2": "value2"}
                tracker.set_tags(tags)
                mock_set_tags.assert_called_once_with(tags)

            # Test end_run
            with patch.object(MockMLflow, "end_run", wraps=MockMLflow.end_run) as mock_end_run:
                tracker.end_run()
                mock_end_run.assert_called_once()

    def test_handle_errors_gracefully(self):
        """Test that MLflowTracker handles errors gracefully."""
        # Use an invalid URI to trigger error handling
        os.environ["MLFLOW_TRACKING_URI"] = "invalid://uri"

        # Should not raise an exception even with invalid URI
        tracker = MLflowTracker()
        assert tracker.mlflow_enabled is False

        # These should not raise exceptions
        tracker.start_run()
        tracker.log_params({"test": "value"})
        tracker.log_metrics({"accuracy": 0.95})
        tracker.end_run()

        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]
