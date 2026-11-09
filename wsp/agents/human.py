"""
A human agent class used for testing and development.

The purpose of this is to give developers a sense of the working environment
from an agent's perspective.
"""

from typing import Any

from wsp.log import RunLog
from wsp.tools import Toolbox

from .base import BaseAgent

PROMPT = """
You are the agent! 

Use the tools in the Toolbox to update and manipulate the model weights.
"""


class HumanAgent(BaseAgent):
    # TODO: Implement this
    def run(
        self, system: str, prompt: str, toolbox: Toolbox, log: RunLog, max_turns: int
    ) -> None:
        """
        You are the agent! Work in your terminal and interact with the tools provided by the Toolbox
        until you write `exit`.
        """
        turn = 0

        while True:
            turn += 1
            command = input(f"(Turn {turn}) Enter command (or `exit` to quit): ")
            if command.strip().lower() == "exit":
                break

    def tool_schemas(self, toolbox: Toolbox) -> Any:
        """
        A human readable description of the tools available in the Toolbox.
        """
        tools = "\n".join(
            f"`{name}`: {tool.description}" for name, tool in toolbox.tools.items()
        )
        return f"--- Tools ---\n\n{tools}\n"
