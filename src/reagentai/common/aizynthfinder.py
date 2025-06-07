import logging

from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.context.config import Configuration

from .utils.cache import RetrosynthesisCache

logger = logging.getLogger(__name__)


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
