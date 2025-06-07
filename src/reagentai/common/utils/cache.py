from aizynthfinder.context.config import Configuration

from src.reagentai.models.retrosynthesis import RouteCollection


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
