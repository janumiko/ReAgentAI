from pathlib import Path
from logging import INFO

INSTRUCTIONS_PATH: str = "src/reagentai/llm/instructions/instructions.txt"
AVAILABLE_LLM_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
]
AIZYNTHFINDER_CONFIG_PATH: str = "data/config.yml"
DEFAULT_LOG_LEVEL: int = INFO
LOG_DIR: Path = Path("logs")
LOG_TO_FILE: bool = True
