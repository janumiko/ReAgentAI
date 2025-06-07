from typing import Protocol

from aizynthfinder.aizynthfinder import AiZynthFinder


class HasAiZynthFinder(Protocol):
    """Protocol for any dependencies that include an AiZynthFinder instance."""

    aizynth_finder: AiZynthFinder
