"""
Tools the agent can call, defined once and independent of the LLM provider.

A tool is a name, a description, a JSON schema for its arguments, and a
handler. Handlers receive the session, the tool's budget, and the arguments,
and return the text the agent sees.
"""

from typing import Any

from wsp.session import Session

from .models import Tool, ToolError, ToolResult


class Budget:
    """
    Counts how much of a tool has been used. `limit=None` means unlimited.
    """

    def __init__(self, limit: int | None):
        self.limit = limit
        self.used = 0

    def __str__(self) -> str:
        return (
            f"{self.used} used"
            if self.limit is None
            else f"{self.used} of {self.limit} used"
        )

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else self.limit - self.used

    def spend(self, amount: int = 1) -> None:
        if self.limit is not None and self.used + amount > self.limit:
            raise ToolError(
                f"Budget exhausted: {self.remaining} of {self.limit} left, {amount} requested."
            )
        self.used += amount


class Toolbox:
    """
    The tools for one run: their budgets, and the cap on how much text the agent gets back.
    """

    def __init__(self, tools: list[Tool], session: Session, max_output_chars: int):
        self.tools = {tool.name: tool for tool in tools}
        self.budgets = {tool.name: Budget(tool.limit) for tool in tools}
        self.session = session
        self.max_output_chars = max_output_chars

    # -------------------------------------------------------------------------------- public

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        Run a tool. The call and its result are logged as two events, so
        anything the tool logs while it runs (an evaluation, a report) sits
        between them, and the times show how long it took.
        """
        self.session.log.event("tool_call", name=name, arguments=arguments)

        if name not in self.tools:
            result = ToolResult(f"Unknown tool '{name}'.", is_error=True)
        else:
            result = self._call(self.tools[name], arguments)

        self.session.log.event(
            "tool_result", name=name, output=result.text, is_error=result.is_error
        )
        return result

    @property
    def finished(self) -> bool:
        """
        True once the agent has called `finish`. Agents stop when it is.
        """
        return self.session.finished

    def usage(self) -> dict[str, str]:

        return {name: str(budget) for name, budget in self.budgets.items()}

    # -------------------------------------------------------------------------------- private

    def _call(self, tool: Tool, arguments: dict[str, Any]) -> ToolResult:
        try:
            text = tool.handler(self.session, self.budgets[tool.name], **arguments)
            return ToolResult(self._truncate(text))
        except Exception as error:
            return ToolResult(f"{type(error).__name__}: {error}", is_error=True)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text

        note = (
            f"\n\n[Output truncated: showing {self.max_output_chars:,} of {len(text):,} characters. "
            "Print a smaller slice or a summary instead.]"
        )
        return text[: self.max_output_chars] + note
