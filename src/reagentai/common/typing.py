from typing import Protocol

from aizynthfinder.aizynthfinder import AiZynthFinder

ChatMessage = dict[str, str | dict[str, str]]
ChatHistory = list[ChatMessage]


class HasAiZynthFinder(Protocol):
    """Protocol for any dependencies that include an AiZynthFinder instance."""

    aizynth_finder: AiZynthFinder
