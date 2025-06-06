from typing import Any, List, Optional
from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.context.config import Configuration
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class FullRetrosynthesisData(BaseModel):
    all_routes: List[dict[str, Any]]
    statistics: dict[str, Any]


class RetrosynthesisResult(BaseModel):
    selected_route_details: dict[str, Any]
    overall_search_statistics: dict[str, Any]

    # You'll likely need to redefine __str__ and __repr__
    # as Pydantic's default repr is comprehensive but can be long.
    def __str__(self) -> str:
        route_info = (
            f"  Number of steps (top route): {self.overall_search_statistics.get('number_of_steps', 'N/A')}\n"
            f"  Top score: {self.overall_search_statistics.get('top_score', 'N/A'):.4f}\n"
            f"  Precursors in stock (top route): {self.overall_search_statistics.get('precursors_in_stock', 'N/A')}\n"
            f"  Total routes found: {self.overall_search_statistics.get('number_of_routes', 'N/A')}"
        )
        return (
            f"Retrosynthesis Analysis Summary:\n{route_info}\n"
            f"\nSelected Route Details:\n{self.selected_route_details}"
        )


class RetrosynthesisCache:
    routes_cache: dict[str, FullRetrosynthesisData] = {}

    @classmethod
    def add(cls, target_smile: str, data: FullRetrosynthesisData):
        cls.routes_cache[target_smile] = data

    @classmethod
    def get(cls, target_smile: str) -> FullRetrosynthesisData | None:
        return cls.routes_cache.get(target_smile)

    @classmethod
    def clear(cls):
        cls.routes_cache.clear()


# --- Global variable to hold the AiZynthFinder instance ---
_global_aizynthfinder_instance: Optional[AiZynthFinder] = None
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


def perform_retrosynthesis(
    target_smile: str, route_index: int = 0
) -> RetrosynthesisResult:
    """
    Perform a retrosynthetic tree search for the given target molecule.
    Accesses a pre-initialized global AiZynthFinder instance.
    Caches all found routes, but returns only the specified route and overall statistics.

    Args:
        target_smile (str): SMILES representation of the target molecule.
        route_index (int): Index of the route to return from the found routes.
                           Defaults to 0 (top-scoring route).

    Returns:
        RetrosynthesisResult: Result containing the selected route details and overall search statistics.

    Raises:
        ValueError: If no routes are found or route_index is out of bounds.
        RuntimeError: If AiZynthFinder has not been initialized.
    """
    global _global_aizynthfinder_instance

    if _global_aizynthfinder_instance is None:
        raise RuntimeError(
            "AiZynthFinder has not been initialized. Call 'initialize_aizynthfinder_globally' first."
        )

    # 1. Check cache first
    cached_full_data = RetrosynthesisCache.get(target_smile)
    if cached_full_data:
        logger.info(
            f"Retrieving full retrosynthesis data for {target_smile} from cache."
        )

        # Validate route_index against cached data
        if not cached_full_data.all_routes:
            raise ValueError(
                f"Cached data found, but no routes present for target: {target_smile}"
            )
        if route_index >= len(cached_full_data.all_routes):
            raise ValueError(
                f"Route index {route_index} is out of bounds for cached data. "
                f"Only {len(cached_full_data.all_routes)} routes found previously."
            )

        # Extract the specific route to return
        selected_route_dict = cached_full_data.all_routes[route_index]
        return RetrosynthesisResult(
            selected_route_details=selected_route_dict,
            overall_search_statistics=cached_full_data.statistics,
        )

    # 2. If not in cache, perform search using the global instance
    logger.info(f"Performing retrosynthesis for {target_smile}...")
    _global_aizynthfinder_instance.target_smiles = target_smile
    _global_aizynthfinder_instance.tree_search()
    _global_aizynthfinder_instance.build_routes()  # This populates finder.routes

    # 3. Extract all routes and statistics
    routes_dicts = _global_aizynthfinder_instance.routes.dict_with_scores()
    statistics = _global_aizynthfinder_instance.extract_statistics()

    if not routes_dicts:  # Check if any routes were found
        raise ValueError(f"No retrosynthesis routes found for target: {target_smile}")

    if route_index >= len(routes_dicts):
        raise ValueError(
            f"Route index {route_index} is out of bounds. Only {len(routes_dicts)} routes found."
        )

    # 4. Create full data object and add to cache
    full_data_to_cache = FullRetrosynthesisData(
        all_routes=routes_dicts,
        statistics=statistics,
    )
    RetrosynthesisCache.add(target_smile, full_data_to_cache)

    # 5. Extract the specific route for return to the agent
    selected_route_dict = routes_dicts[route_index]
    result_for_agent = RetrosynthesisResult(
        selected_route_details=selected_route_dict,
        overall_search_statistics=statistics,
    )

    logger.info(f"Retrosynthesis for {target_smile} completed and cached.")
    return result_for_agent
