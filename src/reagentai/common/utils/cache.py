from aizynthfinder.context.config import Configuration

from src.reagentai.models.retrosynthesis import RouteCollection


class RetrosynthesisCache:
    """
    A cache for storing retrosynthesis routes based on target SMILES strings.
    This class provides methods to add, retrieve, and clear cached routes.
    It also maintains a configuration for the AiZynthFinder instance used in retrosynthesis.
    """

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
