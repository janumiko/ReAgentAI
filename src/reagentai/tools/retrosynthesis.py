import logging
from typing import Protocol

from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.analysis.utils import RouteSelectionArguments
from aizynthfinder.context.config import Configuration
from pydantic_ai import RunContext

from src.reagentai.models.retrosynthesis import RouteCollection

from .helpers import parse_route_dict

logger = logging.getLogger(__name__)


class HasAiZynthFinder(Protocol):
    """Protocol for any dependencies that include an AiZynthFinder instance."""

    aizynth_finder: AiZynthFinder


def initialize_aizynthfinder(
    config_path: str, stock: str, expansion_policy: str, filter_policy: str
) -> AiZynthFinder:
    """
    Initializes the AiZynthFinder instance with the given configuration.

    Args:
        config_path (str): Path to the configuration file for AiZynthFinder.
        stock (str): Stock source to use (e.g., "zinc").
        expansion_policy (str): Expansion policy to use (e.g., "uspto").
        filter_policy (str): Filter policy to use (e.g., "uspto").

    Returns:
        AiZynthFinder: An initialized instance of AiZynthFinder.
    """
    config = Configuration.from_file(config_path)
    finder = AiZynthFinder(
        configfile=config_path,
    )
    finder.stock.select(stock)
    finder.expansion_policy.select(expansion_policy)
    finder.filter_policy.select(filter_policy)

    if RetrosynthesisCache.finder_config is None:
        RetrosynthesisCache.finder_config = config
    elif RetrosynthesisCache.finder_config != config:
        logger.warning("Configuration has changed since the last initialization. Clearing cache.")
        RetrosynthesisCache.clear()
        RetrosynthesisCache.finder_config = config

    return finder


class RetrosynthesisCache:
    routes_cache: dict[str, RouteCollection] = {}
    finder_config: Configuration | None = None

    @classmethod
    def add(cls, target_smile: str, data: RouteCollection):
        cls.routes_cache[target_smile] = data

    @classmethod
    def get(cls, target_smile: str) -> RouteCollection | None:
        return cls.routes_cache.get(target_smile)

    @classmethod
    def clear(cls):
        cls.routes_cache.clear()


def perform_retrosynthesis(
    ctx: RunContext[HasAiZynthFinder], target_smile: str
) -> RouteCollection:
    """
    Performs retrosynthesis for a given target SMILES string using the AiZynthFinder instance.

    Args:
        ctx (RunContext[HasAiZynthFinder]): The run context containing the AiZynthFinder instance.
        target_smile (str): The target SMILES string for retrosynthesis.

    Returns:
        RouteCollection: A collection of retrosynthesis routes found for the target SMILES.

    Raises:
        ValueError: If no retrosynthesis routes are found for the target SMILES.
    """

    finder = ctx.deps.aizynth_finder

    # 1. Check if the target SMILES is already in cache
    cached_routes = RetrosynthesisCache.get(target_smile)
    if cached_routes is not None:
        logger.info(f"Retrieving full retrosynthesis data for {target_smile} from cache.")
        return cached_routes

    # 2. If not in cache, perform search using the global instance
    logger.info(f"Performing retrosynthesis for {target_smile}...")
    selection_args = RouteSelectionArguments(nmin=5, nmax=25)

    finder.target_smiles = target_smile
    finder.tree_search()
    finder.build_routes(selection_args)
    finder.routes.compute_scores(*finder.scorers.objects())

    # 3. Extract all routes and statistics
    routes = finder.routes.dict_with_scores()

    if not routes:  # Check if any routes were found
        raise ValueError(f"No retrosynthesis routes found for target: {target_smile}")

    # 4. Convert to RouteCollection
    routes = [parse_route_dict(route) for route in routes]
    route_collection = RouteCollection(routes=routes, n_routes=len(routes))
    logger.info(f"Found {len(route_collection)} retrosynthesis routes for {target_smile}.")

    # 5. Cache the result
    RetrosynthesisCache.add(target_smile, route_collection)

    return route_collection
