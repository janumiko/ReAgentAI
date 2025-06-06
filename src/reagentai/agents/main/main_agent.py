from pydantic_ai import Tool

from src.reagentai.common.client import LLMClient
from src.reagentai.tools.helpers import create_tool
from src.reagentai.tools.retrosynthesis import perform_retrosynthesis
from src.reagentai.tools.smiles import is_valid_smiles

MAIN_AGENT_INSTRUCTIONS_PATH: str = "src/reagentai/agents/main/instructions.txt"
MAIN_AGENT_MODEL: str = "google-gla:gemini-2.0-flash"


def create_main_agent() -> LLMClient:
    """
    Creates and returns the main agent client with the specified model and instructions.

    Returns:
        LLMClient: An instance of LLMClient configured with the main agent's model and instructions.
    """

    with open(MAIN_AGENT_INSTRUCTIONS_PATH) as instructions_file:
        instructions = instructions_file.read()

    tools = map(create_tool, [perform_retrosynthesis, is_valid_smiles])

    llm_client = LLMClient(model_name=MAIN_AGENT_MODEL, tools=tools, instructions=instructions)

    return llm_client
