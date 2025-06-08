from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging

from aizynthfinder.aizynthfinder import AiZynthFinder
from pydantic_ai import Agent, Tool, result
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.messages import UserPromptPart

from src.reagentai.common.aizynthfinder import initialize_aizynthfinder
from src.reagentai.constants import AIZYNTHFINDER_CONFIG_PATH
from src.reagentai.tools.image import route_to_image, smiles_to_image
from src.reagentai.tools.pubchem import (
    get_compound_info,
    get_name_from_smiles,
    get_smiles_from_name,
)
from src.reagentai.tools.retrosynthesis import perform_retrosynthesis
from src.reagentai.tools.smiles import find_similar_molecules, is_valid_smiles

logger = logging.getLogger(__name__)

MAIN_AGENT_INSTRUCTIONS_PATH: str = "src/reagentai/agents/main/instructions.txt"
MAIN_AGENT_MODEL: str = "google-gla:gemini-2.0-flash"


@dataclass
class MainAgentDependencyTypes:
    """
    Defines the dependencies required by the main agent.
    """

    aizynth_finder: AiZynthFinder


class MainAgent:
    def __init__(
        self,
        model_name: str,
        instructions: str,
        tools: list[Tool],
        dependency_types: type[MainAgentDependencyTypes],
        dependencies: MainAgentDependencyTypes,
        output_type: type[str],
    ):
        """
        Initializes the MainAgent with the specified model, instructions, tools, and dependencies.

        Args:
            model_name (str): The name of the language model to use.
            instructions (str): Instructions for the agent.
            tools (list[Tool]): List of tools that the agent can use.
            dependency_types (type[MainAgentDependencyTypes]): The type of dependencies required by the agent.
            dependencies (MainAgentDependencyTypes): The actual dependencies to be used by the agent.
            output_type (type[str]): The type of output expected from the agent.
        """

        self.model_name = model_name
        self.instructions = instructions
        self.tools = tools
        self.dependency_types = dependency_types
        self.dependencies = dependencies
        self.output_type = output_type

        self.message_history = None
        self.usage = None

        self._agent = self._create_agent()
        logger.info(f"MainAgent initialized with model: {model_name}")

    def _create_agent(self) -> Agent[MainAgentDependencyTypes, str]:
        """
        Creates an Agent instance with the specified model and instructions.

        This method uses the self attributes to configure the Agent.

        Returns:
            Agent[MainAgentDependencyTypes, str]: An instance of the Agent configured with the main agent's model and instructions.
        """

        return Agent(
            self.model_name,
            tools=self.tools,
            instructions=self.instructions,
            deps_type=self.dependency_types,
            output_type=self.output_type,
        )

    def remove_last_messages(self, remove_user_prompt: bool = True):
        """
        Removes the last messages from the agent's message history.
        Args:
            remove_user_prompt (bool): If True, removes the last user prompt as well.
        """
        while self.message_history and not any(
            isinstance(part, UserPromptPart) for part in self.message_history[-1].parts
        ):
            self.message_history.pop()

        if remove_user_prompt and self.message_history:
            self.message_history.pop()

        logger.info("MainAgent last messages removed from history.")

    def set_model(self, model_name: str):
        """
        Sets the model for the Agent.
        Args:
            model_name (str): The name of the new language model to use.
        """

        self.model_name = model_name
        self._agent = self._create_agent()
        logger.info(f"MainAgent model set to: {model_name}")

    def get_total_token_usage(self) -> int:
        """
        Returns the total number of tokens used by the agent.

        Returns:
            int: The total number of tokens used by the agent.
        """
        if self.usage:
            return self.usage.total_tokens
        else:
            return 0

    def clear_history(self):
        """
        Clears the chat history of the agent.
        """

        logger.info("MainAgent chat history cleared.")
        self.message_history = None
        self.usage = None

    @asynccontextmanager
    async def run_stream(self, user_query: str) -> AsyncIterator[result.StreamedRunResult]:
        """
        Streams the response from the agent asynchronously.

        Args:
            user_query (str): The user's query to the agent.
            chat_history (ChatHistory): The current chat history.

        Returns:
            AsyncGenerator: An asynchronous generator yielding messages from the agent.
        """
        if not user_query:
            logger.warning("Empty user query received.")

        async with self._agent.run_stream(
            user_query,
            message_history=self.message_history,
            deps=self.dependencies,
        ) as result:
            yield result

            self.message_history = result.all_messages()
            self.usage = result.usage()

    def run(self, user_query: str) -> AgentRunResult:
        """
        Runs the agent with the given user query and returns the result.

        Args:
            user_query (str): The user's query to the agent.

        Returns:
            AgentRunResult: The result of the agent's run, including the output and message history.
        """
        if not user_query:
            logger.warning("Empty user query received.")

        result: AgentRunResult = self._agent.run_sync(
            user_query,
            message_history=self.message_history,
            deps=self.dependencies,
        )

        self.message_history = result.all_messages()
        self.usage = result.usage()

        return result


def create_main_agent() -> MainAgent:
    """
    Creates and returns the main agent client with the specified model and instructions.

    Returns:
        MainAgent: An instance of MainAgent configured with the main agent's model and instructions.
    """

    # Load instructions
    with open(MAIN_AGENT_INSTRUCTIONS_PATH) as instructions_file:
        instructions = instructions_file.read()

    tools = [
        Tool(perform_retrosynthesis, takes_ctx=True),
        Tool(is_valid_smiles),
        Tool(smiles_to_image),
        Tool(route_to_image),
        Tool(find_similar_molecules),
        Tool(get_smiles_from_name),
        Tool(get_compound_info),
        Tool(get_name_from_smiles),
        duckduckgo_search_tool(),
    ]

    aizynth_finder = initialize_aizynthfinder(
        config_path=AIZYNTHFINDER_CONFIG_PATH,
        stock="zinc",
        expansion_policy="uspto",
        filter_policy="uspto",
    )

    # Initialize the MainAgent client
    main_agent = MainAgent(
        model_name=MAIN_AGENT_MODEL,
        tools=tools,
        instructions=instructions,
        dependency_types=MainAgentDependencyTypes,
        dependencies=MainAgentDependencyTypes(aizynth_finder=aizynth_finder),
        output_type=str,
    )

    return main_agent
