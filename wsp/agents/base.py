from abc import ABC, abstractmethod
from typing import Any

from wsp.log import RunLog
from wsp.tools import Toolbox


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(
        self, system: str, prompt: str, toolbox: Toolbox, log: RunLog, max_turns: int
    ) -> None:
        """
        Work until the model stops calling tools, or ``max_turns`` is reached.
        """

    @abstractmethod
    def tool_schemas(self, toolbox: Toolbox) -> Any:
        """
        Convert the tools defined in the Toolbox to a format that the LLM client accepts.
        """
