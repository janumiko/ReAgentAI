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
    "Suggest a retrosynthesis for Ibuprofen. Show all molecule images from the first route.",
    "Show in the image the best synthesis route of COc1cccc(OC(=O)/C=C/c2cc(OC)c(OC)c(OC)c2)c1 based on score.",
    "Find molecules similar to Aspirin (O=C(C)Oc1ccccc1C(=O)O) from the following list: Paracetamol (CC(=O)Nc1ccc(O)cc1), Ibuprofen (CC(C)Cc1ccc(C(C)C(=O)O)cc1), Naproxen (COc1ccc2cc(C(C)C(=O)O)ccc2c1). Show the top 2.",
]

DEFAULT_LOG_LEVEL: int = INFO
LOG_DIR: Path = Path("logs")
LOG_TO_FILE: bool = True

APP_CSS: str = """
    .contain { display: flex !important; flex-direction: column !important; }
    #chatbot_display { flex-grow: 1 !important; overflow: auto !important;}
    #col { height: calc(95vh - 112px - 16px) !important; }
    #logo_container {
        height: 5vh !important;
        display: flex;
        justify-content: left;
        align-items: center;
    }
"""
