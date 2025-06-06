import logging

from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.context.config import Configuration
from aizynthfinder.analysis.utils import RouteSelectionArguments

from src.reagentai.models.retrosynthesis import RouteCollection

from .helpers import parse_route_dict

logger = logging.getLogger(__name__)


# --- Global variable to hold the AiZynthFinder instance ---
_global_aizynthfinder_instance: AiZynthFinder | None = None
_current_finder_config: Configuration = {}  # To track if config changed and clear cache


def initialize_aizynthfinder_globally(
    config_path: str, stock: str, expansion_policy: str, filter_policy: str
):
    """
    Initializes the global AiZynthFinder instance.
    This function should be called ONCE at application startup.
    """
    global _global_aizynthfinder_instance, _current_finder_config

    new_config = Configuration.from_file(config_path)

    # Only re-initialize and clear cache if config has changed
    if _global_aizynthfinder_instance is None or _current_finder_config != new_config:
        logger.info(f"Initializing AiZynthFinder with new configuration: {new_config}")
        _global_aizynthfinder_instance = AiZynthFinder(configfile=config_path)

        _global_aizynthfinder_instance.stock.select(stock)
        _global_aizynthfinder_instance.expansion_policy.select(expansion_policy)
        _global_aizynthfinder_instance.filter_policy.select(filter_policy)

        # Crucial for cache invalidation:
        # When a new finder object with potentially different configuration is created,
        # the existing cache results are no longer valid, so we clear it.
        RetrosynthesisCache.clear()
        _current_finder_config = new_config
    else:
        logger.info(
            "AiZynthFinder already initialized with current configuration. Skipping re-initialization."
        )


class RetrosynthesisCache:
    routes_cache: dict[str, RouteCollection] = {}

    @classmethod
    def add(cls, target_smile: str, data: RouteCollection):
        cls.routes_cache[target_smile] = data

    @classmethod
    def get(cls, target_smile: str) -> RouteCollection | None:
        return cls.routes_cache.get(target_smile)

    @classmethod
    def clear(cls):
        cls.routes_cache.clear()


def perform_retrosynthesis(target_smile: str) -> RouteCollection:
    """
    Performs retrosynthetic analysis for a given target molecule represented by a SMILES string.

    This function uses a globally initialized AiZynthFinder instance to search for possible retrosynthetic routes
    to synthesize the target molecule. It raises an error if the AiZynthFinder instance is not initialized or if
    no routes are found. The resulting routes are parsed and returned as a RouteCollection object.

    Args:
        target_smile (str): The SMILES string of the target molecule for which retrosynthesis is to be performed.

    Returns:
        RouteCollection: A collection of retrosynthetic routes found for the target molecule.

    Raises:
        RuntimeError: If the global AiZynthFinder instance has not been initialized.
        ValueError: If no retrosynthetic routes are found for the target molecule.
    """
    global _global_aizynthfinder_instance

    if _global_aizynthfinder_instance is None:
        raise RuntimeError(
            "AiZynthFinder has not been initialized. Call 'initialize_aizynthfinder_globally' first."
        )

    # 1. Check if the target SMILES is already in cache
    cached_routes = RetrosynthesisCache.get(target_smile)
    if cached_routes is not None:
        logger.info(f"Retrieving full retrosynthesis data for {target_smile} from cache.")
        return cached_routes

    # 2. If not in cache, perform search using the global instance
    logger.info(f"Performing retrosynthesis for {target_smile}...")
    selection_args = RouteSelectionArguments(nmin=1, nmax=10)

    _global_aizynthfinder_instance.target_smiles = target_smile
    _global_aizynthfinder_instance.tree_search()
    _global_aizynthfinder_instance.build_routes(selection_args)
    _global_aizynthfinder_instance.routes.compute_scores(
        *_global_aizynthfinder_instance.scorers.objects()
    )

    # 3. Extract all routes and statistics
    routes = _global_aizynthfinder_instance.routes.dict_with_scores()

    if not routes:  # Check if any routes were found
        raise ValueError(f"No retrosynthesis routes found for target: {target_smile}")

    # 4. Convert to RouteCollection
    routes = [parse_route_dict(route) for route in routes]
    route_collection = RouteCollection(routes=routes, n_reactions=len(routes))
    logger.info(f"Found {len(route_collection)} retrosynthesis routes for {target_smile}.")

    # 5. Cache the result
    RetrosynthesisCache.add(target_smile, route_collection)

    return route_collection
