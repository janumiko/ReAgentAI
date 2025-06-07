import logging
from typing import Protocol

from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.analysis.utils import RouteSelectionArguments
from aizynthfinder.context.config import Configuration
from pydantic_ai import RunContext

from src.reagentai.common.mlflow_tracking import MLflowTracker
from src.reagentai.models.retrosynthesis import RouteCollection

from .helpers import parse_route_dict

logger = logging.getLogger(__name__)
mlflow_tracker = MLflowTracker(experiment_name="retrosynthesis_experiments")


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
        logger.warning(
            "Configuration has changed since the last initialization. Clearing cache."
        )
        RetrosynthesisCache.clear()
        RetrosynthesisCache.finder_config = config

    # Log initialization parameters with MLflow
    run_id = mlflow_tracker.start_run("aizynthfinder_initialization")
    if run_id:
        mlflow_tracker.log_params(
            {
                "stock": stock,
                "expansion_policy": expansion_policy,
                "filter_policy": filter_policy,
                "config_path": config_path,
            }
        )
        mlflow_tracker.end_run()

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

    # Start MLflow run for this retrosynthesis attempt
    run_name = f"retrosynthesis_{target_smile[:10]}"
    mlflow_tracker.start_run(run_name)
    mlflow_tracker.log_params({"target_smiles": target_smile})

    # 1. Check if the target SMILES is already in cache
    cached_routes = RetrosynthesisCache.get(target_smile)
    if cached_routes is not None:
        logger.info(f"Retrieving full retrosynthesis data for {target_smile} from cache.")
        mlflow_tracker.log_params({"cache_hit": True})
        mlflow_tracker.end_run()
        return cached_routes

    # 2. If not in cache, perform search using the global instance
    logger.info(f"Performing retrosynthesis for {target_smile}...")
    mlflow_tracker.log_params({"cache_hit": False})
    selection_args = RouteSelectionArguments(nmin=5, nmax=25)
    mlflow_tracker.log_params(
        {"selection_nmin": selection_args.nmin, "selection_nmax": selection_args.nmax}
    )

    try:
        # Track start time for performance metrics
        import time

        start_time = time.time()

        finder.target_smiles = target_smile
        finder.tree_search()

        search_time = time.time() - start_time
        mlflow_tracker.log_metrics({"search_time_seconds": search_time})

        finder.build_routes(selection_args)
        finder.routes.compute_scores(*finder.scorers.objects())

        # 3. Extract all routes and statistics
        routes = finder.routes.dict_with_scores()

        if not routes:  # Check if any routes were found
            mlflow_tracker.log_params({"routes_found": False})
            mlflow_tracker.end_run()
            raise ValueError(f"No retrosynthesis routes found for target: {target_smile}")

        # 4. Convert to RouteCollection
        routes = [parse_route_dict(route) for route in routes]
        route_collection = RouteCollection(routes=routes, n_routes=len(routes))
        logger.info(f"Found {len(route_collection)} retrosynthesis routes for {target_smile}.")

        # Log metrics about the found routes
        mlflow_tracker.log_metrics(
            {
                "num_routes": len(routes),
                "best_state_score": routes[0].score.state_score if routes else 0,
                "avg_reactions_per_route": sum(r.score.n_reactions for r in routes) / len(routes)
                if routes
                else 0,
            }
        )

        # Generate and log visualization for the best route if routes exist
        if routes and hasattr(finder, "plot_route"):
            try:
                from src.reagentai.tools.smiles import smiles_to_image

                best_route_img = smiles_to_image(target_smile)
                mlflow_tracker.log_artifact(best_route_img)
            except Exception as e:
                logger.warning(f"Failed to generate route visualization: {e}")

        # 5. Cache the result
        RetrosynthesisCache.add(target_smile, route_collection)

        mlflow_tracker.log_params({"routes_found": True})
        mlflow_tracker.end_run()
        return route_collection

    except Exception as e:
        # Log any errors that occur during retrosynthesis
        mlflow_tracker.log_params({"error": str(e), "error_type": type(e).__name__})
        mlflow_tracker.end_run()
        raise
