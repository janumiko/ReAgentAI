from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol

from aizynthfinder.aizynthfinder import AiZynthFinder
from pydantic_ai import Agent, RunContext, Tool

from src.reagentai.common.aizynthfinder import initialize_aizynthfinder
from src.reagentai.constants import AIZYNTHFINDER_CONFIG_PATH
from src.reagentai.models.retrosynthesis import RouteCollection
from src.reagentai.tools.retrosynthesis import perform_retrosynthesis
from src.reagentai.tools.smiles import is_valid_smiles

logger = logging.getLogger(__name__)

RETROSYNTH_AGENT_INSTRUCTIONS_PATH: str = "src/reagentai/agents/retrosynth_agent/instructions.txt"
RETROSYNTH_AGENT_MODEL: str = "google-gla:gemini-2.0-flash"


class HasRetrosynthAgent(Protocol):
    retrosynth_agent: RetrosynthAgent


@dataclass
class RetrosynthAgentDependencyTypes:
    """
    Dependency types for the RetrosynthAgent.
    This class defines the
    dependencies required by the RetrosynthAgent to perform retrosynthesis tasks.
    """

    aizynth_finder: AiZynthFinder


async def delegate_to_retroagent(
    ctx: RunContext[HasRetrosynthAgent], query: str
) -> RouteCollection:
    """
    Delegates a query to the retrosynthesis agent.

    Use this agent to perform retrosynthesis or SMILE validation tasks by providing a query.
    This agent can:
     - Perform retrosynthesis for a given target SMILES string using external libraries and tools.
     - Validate SMILES strings.

    Args:
        ctx (RunContext[HasRetrosynthAgent]): The run context containing the retrosynthesis agent.
        query (str): The query string for the retrosynthesis task.

    Returns:
        RouteCollection: A collection of retrosynthesis routes found by the agent.
    """

    retrosynth_agent = ctx.deps.retrosynth_agent
    output = await retrosynth_agent.respond(query)
    return output


class RetrosynthAgent:
    """
    The RetrosynthAgent class encapsulates the functionality of the retrosynthesis agent.
    It provides methods to perform retrosynthesis, validate SMILES strings, and generate images
    from SMILES or reaction routes.
    """

    def __init__(
        self,
        model_name: str,
        instructions: str,
        tools: list[Tool],
        dependency_types: type[RetrosynthAgentDependencyTypes],
        dependencies: RetrosynthAgentDependencyTypes,
        output_type: str,
    ):
        """
        Initializes the RetrosynthAgent with the specified model, instructions, tools, and dependencies.

        Args:
            model_name (str): The name of the language model to use.
            instructions (str): Instructions for the agent.
            tools (list[Tool]): A list of tools that the agent can use.
            dependency_types type[RetrosynthAgentDependencyTypes]: Types of dependencies required by the agent.
            dependencies (RetrosynthAgentDependencyTypes): Actual dependencies to be used by the agent.
            output_type str : The type of output expected from the agent, typically a RouteCollection.
        """

        self.model_name = model_name
        self.instructions = instructions
        self.tools = tools
        self.dependency_types = dependency_types
        self.dependencies = dependencies
        self.output_type = output_type
        self.result_history = None
        self._agent = self._create_agent()
        logger.info(f"""RetrosynthAgent initialized with model: {model_name}""")

    def _create_agent(self) -> Agent[RetrosynthAgentDependencyTypes, str]:
        """
        Creates an Agent instance with the specified model and instructions.

        This method uses the self attributes to configure the Agent.

        Returns:
            Agent[RetrosynthAgentDependencyTypes, str]: An instance of the Agent configured with the retrosynthesis agent's model and instructions.
        """

        return Agent(
            self.model_name,
            tools=self.tools,
            instructions=self.instructions,
            deps_type=self.dependency_types,
            output_type=self.output_type,
        )

    def set_model(self, model_name: str) -> None:
        """
        Sets the model for the Agent.

        Args:
            model_name (str): The name of the new language model to use.
        """

        self.model_name = model_name
        self._agent = self._create_agent()
        logger.info(f"""RetrosynthAgent model set to: {model_name}""")

    def get_token_usage(self) -> int:
        """
        Returns the token usage of the current agent.

        Returns:
            int: The number of tokens used.
        """

        if self.result_history is not None:
            return self.result_history.usage().total_tokens
        else:
            return 0

    def clear_history(self) -> None:
        """
        Clears the chat history of the agent.
        """

        logger.info("RetrosynthAgent chat history cleared.")
        self.result_history = None

    async def respond(self, user_query: str) -> RouteCollection:
        """
        Responds to a user query and updates the chat history asynchronously.

        Args:
            user_query (str): The user's query.

        Returns:
            RouteCollection: The retrosynthesis routes found by the agent in response to the query.
        """

        if self.result_history is not None:
            message_history = self.result_history.all_messages()
        else:
            message_history = None
        result = await self._agent.run(
            user_query,
            history=message_history,
            deps=self.dependencies,
        )
        self.result_history = result
        logger.info(f"""RetrosynthAgent response: {result.output}""")
        return result.output


def create_retrosynth_agent() -> RetrosynthAgent:
    """
    Creates and returns the retrosynthesis agent client with the specified model and instructions.

    Returns:
        RetrosynthAgent: An instance of RetrosynthAgent configured with the retrosynthesis agent's model and instructions.
    """

    logger.info("Creating RetrosynthAgent...")
    with open(RETROSYNTH_AGENT_INSTRUCTIONS_PATH) as instructions_file:
        instructions = instructions_file.read()

    tools = [
        Tool(
            perform_retrosynthesis,
        ),
        Tool(
            is_valid_smiles,
        ),
    ]

    aizynth_finder = initialize_aizynthfinder(
        config_path=AIZYNTHFINDER_CONFIG_PATH,
        stock="zinc",
        expansion_policy="uspto",
        filter_policy="uspto",
    )

    retrosynth_agent = RetrosynthAgent(
        model_name=RETROSYNTH_AGENT_MODEL,
        instructions=instructions,
        tools=tools,
        dependency_types=RetrosynthAgentDependencyTypes,
        dependencies=RetrosynthAgentDependencyTypes(aizynth_finder=aizynth_finder),
        output_type=str,
    )

    return retrosynth_agent
