import logging
import os

from dotenv import load_dotenv

from src.reagentai.agents.main.main_agent import create_main_agent
from src.reagentai.common.mlflow_tracking import MLflowTracker
from src.reagentai.logging import setup_logging
from src.reagentai.ui.app import create_gradio_app

logger = logging.getLogger(__name__)


def start_agent():
    setup_logging()
    load_dotenv()

    # Initialize MLflow tracking
    tracker = MLflowTracker(experiment_name="reagentai_experiments")

    # Start a new run for this application session
    run_id = tracker.start_run(run_name="reagentai_session")
    logger.info(f"MLflow tracking {'enabled' if run_id else 'disabled'}")

    # Log system information and configuration parameters
    if tracker.mlflow_enabled:
        import platform
        import sys

        # Log system info as tags
        tracker.set_tags(
            {
                "python_version": sys.version,
                "platform": platform.platform(),
                "application": "ReagentAI",
            }
        )

        # Log configuration parameters
        tracker.log_params(
            {
                "log_to_file": os.environ.get("LOG_TO_FILE", "True"),
                "app_version": "0.1.0",  # Could be pulled from a version file
            }
        )

    main_agent = create_main_agent()

    # Pass the MLflow tracker to the Gradio app
    app = create_gradio_app(main_agent, mlflow_tracker=tracker)

    # Launch the application
    try:
        app.launch(server_name="0.0.0.0")
    finally:
        # End the MLflow run when the application exits
        if tracker.mlflow_enabled:
            tracker.end_run()
            logger.info("MLflow tracking session ended")
