from logging import INFO
from pathlib import Path

AVAILABLE_LLM_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
]
AIZYNTHFINDER_CONFIG_PATH: str = "data/config.yml"
EXAMPLE_PROMPTS = [
    "Show the SMILES and structure of Caffeine.",
    "Suggest a retrosynthesis for Aspirin. Show the top 3 routes.",
    "Suggest a retrosynthesis for Ibuprofen. Show molecule images from the first route.",
]

DEFAULT_LOG_LEVEL: int = INFO
LOG_DIR: Path = Path("logs")
LOG_TO_FILE: bool = True
