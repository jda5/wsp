from typing import Any

import anthropic

from wsp.log import RunLog
from wsp.tools import Toolbox

from .base import BaseAgent


class ClaudeAgent(BaseAgent):
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        max_tokens: int = 32_000,
    ):
        self.name = model
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    # -------------------------------------------------------------------------------- public

    def run(
        self, system: str, prompt: str, toolbox: Toolbox, log: RunLog, max_turns: int
    ) -> None:
        messages = [{"role": "user", "content": prompt}]

        for turn in range(1, max_turns + 1):
            response = self._send(system, messages, toolbox)
            messages.append({"role": "assistant", "content": response.content})
            self._record(turn, response, log)

            tool_calls = [
                block for block in response.content if block.type == "tool_use"
            ]

            if response.stop_reason != "tool_use" or not tool_calls:
                log.event("agent_stopped", reason=response.stop_reason, turn=turn)
                return

            results = [self._call_tool(call, toolbox) for call in tool_calls]

            if toolbox.finished:
                log.event("agent_stopped", reason="finished", turn=turn)
                return

            # A text block after the tool results; it stays in the history, so
            # the cached prefix is never edited.
            notice = self.turns_left_notice(turn, max_turns, toolbox)
            if notice:
                results.append({"type": "text", "text": notice})
                log.event("notice", turn=turn, text=notice)

            messages.append({"role": "user", "content": results})

        log.event("agent_stopped", reason="max_turns", turn=max_turns)

    def tool_schemas(self, toolbox: Toolbox) -> Any:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in toolbox.tools.values()
        ]

    # -------------------------------------------------------------------------------- private

    def _send(
        self, system: str, messages: list, toolbox: Toolbox
    ) -> anthropic.types.Message:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=self.tool_schemas(toolbox),
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        ) as stream:
            return stream.get_final_message()

    def _call_tool(self, call: anthropic.types.ToolUseBlock, toolbox: Toolbox) -> dict:
        result = toolbox.call(call.name, call.input)  # logged by the Toolbox

        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": result.text,
            "is_error": result.is_error,
        }

    def _record(
        self, turn: int, response: anthropic.types.Message, log: RunLog
    ) -> None:
        for block in response.content:
            if block.type == "text":
                log.event("text", turn=turn, text=block.text)

            elif block.type == "thinking":
                log.event("thinking", turn=turn, text=block.thinking)

        log.event("usage", turn=turn, usage=response.usage.model_dump())
