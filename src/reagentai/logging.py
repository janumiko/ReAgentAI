from datetime import datetime
import logging
from logging import FileHandler
from pathlib import Path

from src.reagentai.constants import DEFAULT_LOG_LEVEL, LOG_DIR, LOG_TO_FILE


def setup_logging(
    log_dir: Path = LOG_DIR,
    log_level: int = DEFAULT_LOG_LEVEL,
    log_to_file: bool = LOG_TO_FILE,
    # Removed: when, interval, backup_count as they are for rotating handlers
):
    """
    Configures application logging with a single log file per run.

    Args:
        log_dir (Path): Directory where log files will be stored.
        log_level (int): Numeric logging level (e.g., logging.INFO, logging.DEBUG).
        log_to_file (bool): Whether to log to a file.
    """

    log_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers from the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Specific logger configuration for aizynthfinder
    aizynthfinder_logger = logging.getLogger("aizynthfinder")
    aizynthfinder_logger.setLevel(logging.INFO)
    aizynthfinder_logger.handlers.clear()  # Clear existing handlers
    aizynthfinder_logger.propagate = True

    if log_to_file:
        # Create a new log file with a timestamp for each application run
        current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = log_dir / f"reagentai_{current_time_str}.log"

        file_handler = FileHandler(
            filename=log_file_path,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    print(
        f"Logging configured: Console output and file '{log_file_path}' "
        f"at level {logging.getLevelName(log_level)}"
    )
