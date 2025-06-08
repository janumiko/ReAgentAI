from logging import INFO
from pathlib import Path

AVAILABLE_LLM_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
]
AIZYNTHFINDER_CONFIG_PATH: str = "data/config.yml"
EXAMPLE_PROMPTS = [
    "Can u tell me the SMILES of Caffeine? And generate an image of the compound.",
    "How to synthesize Aspirin? Can u tell me the best steps to achieve this?",
    "Suggest a retrosynthesis for Ibuprofen. Show all molecule images from the best route.",
    "Find molecules similar to Aspirin. Show the top 5.",
]

DEFAULT_LOG_LEVEL: int = INFO
LOG_DIR: Path = Path("logs")
LOG_TO_FILE: bool = True

APP_CSS: str = """
    .contain { display: flex !important; flex-direction: column !important; }
    #chatbot_display { flex-grow: 1 !important; overflow: auto !important;}
    #tool_display { flex-grow: 1 !important; overflow: auto !important;}
    #col { height: calc(95vh - 112px - 16px) !important; }
    #logo_container {
        height: 5vh !important;
        display: flex;
        justify-content: left;
        align-items: center;
    }
"""
