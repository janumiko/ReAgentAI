import logging

from src.reagentai.logging import setup_logging
from dotenv import load_dotenv
from src.reagentai.aizynth.retrosynthesis import initialize_aizynthfinder_globally
from src.reagentai.llm.client import LLMClient
from src.reagentai.ui.app import get_gradio_app
from src.reagentai.core.registers import get_registered_tools

from src.reagentai.constants import INSTRUCTIONS_PATH, AIZYNTHFINDER_CONFIG_PATH

logger = logging.getLogger(__name__)


def get_instructions(instruction_file_path: str) -> str:
    """
    Read instructions from a file.

    Returns:
        str: The instructions read from the file.
    """

    with open(instruction_file_path, "r") as instructions_file:
        instructions = instructions_file.read()

    return instructions


def start_agent():
    setup_logging()
    load_dotenv()
    initialize_aizynthfinder_globally(
        config_path=AIZYNTHFINDER_CONFIG_PATH,
        stock="zinc",
        expansion_policy="uspto",
        filter_policy="uspto",
    )
    tools = get_registered_tools()
    instructions = get_instructions(INSTRUCTIONS_PATH)
    llm_client = LLMClient(
        model_name="google-gla:gemini-2.0-flash", tools=tools, instructions=instructions
    )
    app = get_gradio_app(llm_client)
    app.launch(server_name="127.0.0.1")
