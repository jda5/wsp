"""
A live view of a run in the terminal.

ConsoleView is a RunLog listener: it receives every event as it is logged and
shows the ones that matter at its level.

    quiet     evaluations, notices, the report and the final test scores
    normal    + the agent's messages, every tool call, and output previews
    verbose   + thinking, full outputs, samples, commits and token usage

The same view replays or follows a log file: python -m wsp.watch <run folder>.
"""

import json
from typing import Literal

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from wsp import views

Level = Literal["quiet", "normal", "verbose"]

LEVELS: dict[str, int] = {"quiet": 0, "normal": 1, "verbose": 2}

# Lines of tool output shown at the normal level.
PREVIEW_LINES = 20


class ConsoleView:
    def __init__(self, level: Level = "normal", console: Console | None = None):
        self.level = LEVELS[level]
        self.console = console or Console(highlight=False)
        self.turn = 0

    def __call__(self, event: dict) -> None:
        kind = event.get("kind", "")
        handler = getattr(self, f"_on_{kind}", self._on_unknown)

        if "turn" in event and kind != "agent_stopped":
            self._mark_turn(event["turn"])

        handler(event)

    # -------------------------------------------------------------------------------- the run

    def _on_start(self, event: dict) -> None:
        self.console.rule(f"[bold]{event['experiment']}[/] · {event['agent']}")

        if self._shows("normal"):
            self.console.print(Text(event["prompt"], style="dim"))

    def _on_end(self, event: dict) -> None:
        self.console.rule("[bold]results")
        self.console.print(f"test before  {_scalars(event['test_before'])}")
        self.console.print(f"test after   [bold]{_scalars(event['test_after'])}[/]")
        self.console.print(
            f"changed      {event['values_changed']:,} of {event['values_total']:,} values, "
            f"{event['commits']} commits"
        )

        if self._shows("normal"):
            usage = ", ".join(f"{name} {used}" for name, used in event["tool_usage"].items())
            self.console.print(Text(f"tools        {usage}", style="dim"))

    def _on_agent_stopped(self, event: dict) -> None:
        self.console.print(f"\n[bold]stopped[/] after turn {event['turn']}: {event['reason']}")

    # -------------------------------------------------------------------------------- the agent

    def _on_text(self, event: dict) -> None:
        if self._shows("normal"):
            self.console.print(Markdown(event["text"]))

    def _on_thinking(self, event: dict) -> None:
        if self._shows("verbose") and event["text"].strip():
            self.console.print(Text(event["text"], style="dim italic"))

    def _on_notice(self, event: dict) -> None:
        self.console.print(Text(event["text"], style="yellow"))

    def _on_report(self, event: dict) -> None:
        self.console.print(Panel(Markdown(event["report"]), title="report", border_style="green"))

    def _on_usage(self, event: dict) -> None:
        if not self._shows("verbose"):
            return

        usage = event["usage"]
        self.console.print(
            Text(
                f"tokens: {usage.get('input_tokens', 0):,} in "
                f"(+{usage.get('cache_read_input_tokens') or 0:,} read from cache, "
                f"{usage.get('cache_creation_input_tokens') or 0:,} written), "
                f"{usage.get('output_tokens', 0):,} out",
                style="dim",
            )
        )

    # -------------------------------------------------------------------------------- tools

    def _on_tool_call(self, event: dict) -> None:
        if not self._shows("normal"):
            return

        name, arguments = event["name"], event["arguments"]

        if name == "python":
            self.console.print("\n[bold cyan]→ python[/]")
            self.console.print(Syntax(arguments["code"], "python", theme="ansi_dark"))
        else:
            details = json.dumps(arguments) if arguments else ""
            self.console.print(f"\n[bold cyan]→ {name}[/] [dim]{details}[/]")

    def _on_tool_result(self, event: dict) -> None:
        if not self._shows("normal"):
            return

        # These results are already shown by the evaluate and report events.
        if event["name"] in ("evaluate", "finish") and not event["is_error"]:
            return

        self._print_output(event["output"], is_error=event["is_error"])

    def _on_tool(self, event: dict) -> None:
        """
        Logs from before tool calls were split into tool_call and tool_result.
        """
        self._on_tool_call(event)
        self._on_tool_result(event)

    def _on_evaluate(self, event: dict) -> None:
        scores = event["scores"]
        self.console.print(f"[bold green]evaluation[/] (commit #{event['commit']}): {_scalars(scores)}")

        if self._shows("verbose"):
            self.console.print(views.render_scores(scores))

    def _on_samples(self, event: dict) -> None:
        if self._shows("verbose"):
            self.console.print(Text(f"samples drawn: {_short_list(event['indices'])}", style="dim"))

    def _on_forward(self, event: dict) -> None:
        if self._shows("verbose"):
            self.console.print(Text(f"forward on: {_short_list(event['indices'])}", style="dim"))

    def _on_commit(self, event: dict) -> None:
        if self._shows("verbose"):
            self.console.print(Text(f"commit #{event['number']}: {event['message']}", style="dim"))

    def _on_reset(self, event: dict) -> None:
        if self._shows("normal"):
            self.console.print(Text(f"reset to {event['to'].replace('_', ' ')}", style="yellow"))

    def _on_final_weights_invalid(self, event: dict) -> None:
        self.console.print(Text(f"final weights were invalid: {event['error']}", style="red"))

    def _on_unknown(self, event: dict) -> None:
        if self._shows("verbose"):
            details = {k: v for k, v in event.items() if k not in ("time", "kind")}
            self.console.print(Text(f"{event.get('kind')}: {details}", style="dim"))

    # -------------------------------------------------------------------------------- helpers

    def _shows(self, level: Level) -> bool:
        return self.level >= LEVELS[level]

    def _mark_turn(self, turn: int) -> None:
        """
        Draw a divider when a new turn starts.
        """
        if turn > self.turn and self._shows("normal"):
            self.console.rule(f"turn {turn}", style="dim")
        self.turn = max(self.turn, turn)

    def _print_output(self, output: str, is_error: bool) -> None:
        lines = output.rstrip().splitlines()

        if not self._shows("verbose") and len(lines) > PREVIEW_LINES:
            hidden = len(lines) - PREVIEW_LINES
            lines = lines[:PREVIEW_LINES] + [f"… {hidden} more lines (use --view verbose)"]

        self.console.print(Text("\n".join(lines), style="red" if is_error else ""))


def _scalars(scores: dict) -> str:
    """
    The single-number metrics, e.g. "accuracy 0.5806".
    """
    numbers = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
    return "  ".join(f"{name} {value:.4f}" for name, value in numbers.items()) or "(no scalar metrics)"


def _short_list(values: list, limit: int = 12) -> str:
    shown = ", ".join(str(v) for v in values[:limit])
    return shown + (f", … ({len(values)} in all)" if len(values) > limit else "")
