"""
You are the agent!

Use it while developing the harness to see exactly what an LLM agent sees:
the same system prompt, the same tools, budgets and output cap, and the same
run log. Every tool call goes through the Toolbox, just as an LLM's would.

    > info()                          a line of numpy runs in the python tool
    > code                            several lines; finish with an empty line
    > evaluate                        tools are called by name...
    > forward 3 17 42                 ...with their arguments after it
    > commit tried a ring template
    > help                            everything else
"""

import json
import readline  # noqa: F401 (gives input() arrow keys and history)
import textwrap
from typing import Any

from wsp.log import RunLog
from wsp.tools import Tool, Toolbox

from .base import BaseAgent

RULE = "-" * 80

HELP = """\
Commands
  help                show this message
  tools               describe every tool and its arguments
  system              print the system prompt again
  prompt              print the task prompt again
  budgets             show how much of each tool you have used
  note <text>         write your reasoning to the run log, as an agent would
  code                enter several lines of code; finish with an empty line
  exit                stop without a report (Ctrl-D works too; Ctrl-C clears
                      the line). To end properly, use `finish <report>`

Calling tools
  <tool> <arguments>  e.g. `evaluate`, `forward 3 17 42`, `reset baseline`,
                      `commit tried a ring template`
  <tool> {json}       any tool, with its arguments as a JSON object

Anything else is run as code in the python tool, e.g. `info()`."""


class HumanAgent(BaseAgent):
    name = "human"

    def __init__(self, show_system_prompt: bool = True):
        self.show_system_prompt = show_system_prompt

    def run(
        self, system: str, prompt: str, toolbox: Toolbox, log: RunLog, max_turns: int
    ) -> None:
        """
        Read commands from the terminal until `finish`, `exit`, Ctrl-D or
        `max_turns`. Each tool call is one turn.
        """
        self._introduce(system, prompt, toolbox)
        turn = 0

        while turn < max_turns and not toolbox.finished:
            try:
                line = input(f"\n[turn {turn + 1}/{max_turns}] > ").strip()
            except KeyboardInterrupt:  # Ctrl-C clears the line, as in a shell
                print()
                continue
            except EOFError:  # Ctrl-D finishes the run
                print()
                break

            if not line:
                continue

            command, _, rest = line.partition(" ")

            match command:
                case "exit" | "quit":
                    break

                case "help":
                    print(HELP)

                case "tools":
                    print(self.tool_schemas(toolbox))

                case "system":
                    print(system)

                case "prompt":
                    print(prompt)

                case "budgets":
                    for name, usage in toolbox.usage().items():
                        print(f"  {name:10} {usage}")

                case "note":
                    log.event("text", turn=turn + 1, text=rest)
                    print("(noted)")

                case "code":
                    turn += 1
                    self._call(toolbox, "python", {"code": _read_block()}, turn, max_turns)

                case _ if command in toolbox.tools:
                    tool = toolbox.tools[command]
                    try:
                        arguments = _parse_arguments(tool, rest.strip())
                    except ValueError as error:
                        print(f"Could not read the arguments: {error}")
                        print(f"Usage: {_usage(tool)}")
                        continue

                    turn += 1
                    self._call(toolbox, command, arguments, turn, max_turns)

                case _:
                    turn += 1
                    self._call(toolbox, "python", {"code": line}, turn, max_turns)

        if toolbox.finished:
            reason = "finished"
        elif turn >= max_turns:
            reason = "max_turns"
        else:
            reason = "human_exit"

        log.event("agent_stopped", reason=reason, turn=turn)
        print(f"\nRun finished after {turn} turns ({reason}).")

    def tool_schemas(self, toolbox: Toolbox) -> str:
        """
        A human-readable description of every tool and how to call it.
        """
        sections = []

        for tool in toolbox.tools.values():
            description = textwrap.fill(tool.description, width=76)
            description = textwrap.indent(description, "    ")
            sections.append(f"  {_usage(tool)}\n{description}")

        return "Tools\n\n" + "\n\n".join(sections)

    # -------------------------------------------------------------------------------- private

    def _introduce(self, system: str, prompt: str, toolbox: Toolbox) -> None:
        print(RULE)
        print("You are the agent. Type `help` for commands.")
        print(RULE)

        if self.show_system_prompt:
            print(f"\n{system}\n")
            print(RULE)

        print(f"\n{self.tool_schemas(toolbox)}\n")
        print(RULE)
        print(f"\n{prompt}")

    def _call(
        self, toolbox: Toolbox, name: str, arguments: dict[str, Any], turn: int, max_turns: int
    ) -> None:
        """
        Call a tool through the Toolbox and print exactly what an agent would
        see, including the reminder to wrap up near the end.
        """
        result = toolbox.call(name, arguments)

        if result.is_error:
            print(f"[error] {result.text}")
        else:
            print(result.text)

        notice = self.turns_left_notice(turn, max_turns, toolbox)
        if notice and not toolbox.finished:
            print(f"\n{notice}")


# -------------------------------------------------------------------------------- private


def _read_block() -> str:
    """
    Read lines until an empty line, like a Python prompt.
    """
    print("(enter code; finish with an empty line)")
    lines = []

    while True:
        try:
            line = input("... ")
        except EOFError:
            break

        if not line.strip():
            break
        lines.append(line)

    return "\n".join(lines)


def _parse_arguments(tool: Tool, text: str) -> dict[str, Any]:
    """
    Turn what was typed after a tool's name into its arguments.

    A JSON object is passed as it is. Otherwise, for a tool with a single
    parameter, the text is read according to that parameter's type:

        string              the text itself      commit tried a ring template
        array of integers   numbers              forward 3 17 42  (or 3,17,42)
        integer, number     one number
    """
    if text.startswith("{"):
        return json.loads(text)

    properties = tool.parameters.get("properties", {})

    if not properties:
        if text:
            raise ValueError(f"`{tool.name}` takes no arguments")
        return {}

    if len(properties) > 1:
        raise ValueError(f"`{tool.name}` takes several arguments; pass a JSON object")

    if not text:
        if tool.parameters.get("required"):
            raise ValueError("an argument is required")
        return {}  # the argument is optional, e.g. `forward` runs every sample

    name, schema = next(iter(properties.items()))

    if "enum" in schema and text not in schema["enum"]:
        raise ValueError(f"must be one of: {', '.join(schema['enum'])}")

    match schema.get("type"):
        case "string":
            return {name: text}

        case "array":
            return {name: [int(item) for item in text.replace(",", " ").split()]}

        case "integer":
            return {name: int(text)}

        case "number":
            return {name: float(text)}

    raise ValueError(f"cannot read a `{schema.get('type')}`; pass a JSON object")


def _usage(tool: Tool) -> str:
    """
    e.g. "forward <indices: integers>" or "evaluate"
    """
    properties = tool.parameters.get("properties", {})
    parts = [tool.name]

    for name, schema in properties.items():
        kind = schema.get("type", "value")

        if kind == "array":
            kind = f"{schema.get('items', {}).get('type', 'value')}s"
        if "enum" in schema:
            kind = " | ".join(schema["enum"])

        parts.append(f"<{name}: {kind}>")

    return " ".join(parts)
