import logging

from aizynthfinder.analysis.utils import RouteSelectionArguments
from pydantic_ai import RunContext

from src.reagentai.common.typing import HasAiZynthFinder
from src.reagentai.common.utils.cache import RetrosynthesisCache
from src.reagentai.common.utils.parse import parse_route_dict
from src.reagentai.models.retrosynthesis import RouteCollection

logger = logging.getLogger(__name__)


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

    logger.info(f"[TASK] [PERFORM_RETROSYNTHESIS] Arguments: target_smile: {target_smile}")
    logger.debug(f"Context: {ctx}")

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
    routes = [parse_route_dict(route, idx) for idx, route in enumerate(routes)]
    route_collection = RouteCollection(routes=routes, n_routes=len(routes))
    logger.info(f"Found {len(route_collection)} retrosynthesis routes for {target_smile}.")

    # 5. Cache the result
    RetrosynthesisCache.add(target_smile, route_collection)

    logger.debug(f"Output perform_retrosynthesis: {route_collection}")

    return route_collection
