from dotenv import load_dotenv
from src.reagentai.llm.client import LLMClient
from src.reagentai.ui.app import get_gradio_app
from src.reagentai.core.registers import get_registered_tools
import logging

logging.basicConfig(level=logging.INFO)
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


def main():
    load_dotenv()
    tools = get_registered_tools()
    llm_client = LLMClient(model_name="google-gla:gemini-2.0-flash", tools=tools)
    app = get_gradio_app(llm_client)
    app.launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()
