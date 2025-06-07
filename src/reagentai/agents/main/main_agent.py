from dataclasses import dataclass

from aizynthfinder.aizynthfinder import AiZynthFinder
from pydantic_ai import Tool

from src.reagentai.common.client import LLMClient
from src.reagentai.common.mlflow_tracking import MLflowTracker
from src.reagentai.constants import AIZYNTHFINDER_CONFIG_PATH
from src.reagentai.tools.retrosynthesis import (
    initialize_aizynthfinder,
    perform_retrosynthesis,
)
from src.reagentai.tools.smiles import is_valid_smiles, smiles_to_image

MAIN_AGENT_INSTRUCTIONS_PATH: str = "src/reagentai/agents/main/instructions.txt"
MAIN_AGENT_MODEL: str = "google-gla:gemini-2.0-flash"

# Initialize MLflow tracker for the main agent
agent_mlflow_tracker = MLflowTracker(experiment_name="main_agent_interactions")


@dataclass
class MainAgentDependencyTypes:
    """
    Defines the dependencies required by the main agent.
    This includes the AiZynthFinder instance for retrosynthesis tasks.
    """

    aizynth_finder: AiZynthFinder


def create_main_agent() -> LLMClient:
    """
    Creates and returns the main agent client with the specified model and instructions.

    Returns:
        LLMClient: An instance of LLMClient configured with the main agent's model and instructions.
    """

    # Load instructions
    with open(MAIN_AGENT_INSTRUCTIONS_PATH) as instructions_file:
        instructions = instructions_file.read()

    tools = [Tool(perform_retrosynthesis), Tool(is_valid_smiles), Tool(smiles_to_image)]

    aizynth_finder = initialize_aizynthfinder(
        config_path=AIZYNTHFINDER_CONFIG_PATH,
        stock="zinc",
        expansion_policy="uspto",
        filter_policy="uspto",
    )

    # Initialize the LLM client
    llm_client = LLMClient(
        model_name=MAIN_AGENT_MODEL,
        tools=tools,
        instructions=instructions,
        dependency_types=MainAgentDependencyTypes,
        dependencies=MainAgentDependencyTypes(aizynth_finder=aizynth_finder),
        mlflow_tracker=agent_mlflow_tracker,
    )

    return llm_client
