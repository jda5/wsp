from abc import ABC, abstractmethod
from typing import Any

from wsp.log import RunLog
from wsp.tools import Toolbox

# Warn the agent when this many turns are left, and again on its last turn.
WARN_AT_TURNS_LEFT = 5


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(
        self, system: str, prompt: str, toolbox: Toolbox, log: RunLog, max_turns: int
    ) -> None:
        """
        Work until the agent calls `finish`, stops calling tools, or
        ``max_turns`` is reached.
        """

    @abstractmethod
    def tool_schemas(self, toolbox: Toolbox) -> Any:
        """
        Convert the tools defined in the Toolbox to a format that the LLM client accepts.
        """

    @staticmethod
    def turns_left_notice(turns_used: int, max_turns: int, toolbox: Toolbox) -> str | None:
        """
        A reminder to wrap up, or None. Agents add it after a turn's tool results.
        """
        turns_left = max_turns - turns_used

        if "finish" in toolbox.tools:
            wrap_up = "make sure W holds your best weights, then call `finish` with your report"
        else:
            wrap_up = "make sure W holds your best weights, then reply with your report"

        if turns_left == WARN_AT_TURNS_LEFT:
            return f"[{turns_left} turns left. Start wrapping up: {wrap_up}.]"

        if turns_left == 1:
            return f"[This is your last turn: {wrap_up}.]"

        return None
