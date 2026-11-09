from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """
    An error the agent should see and can recover from.
    """


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    limit: int | None = None


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool = False
