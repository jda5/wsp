import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

# Never shortened
KEEP_IN_FULL = {"code", "system", "prompt", "report"}

# Receives each event as it is written, e.g. a ConsoleView showing the run live.
Listener = Callable[[dict], None]


class RunLog:
    def __init__(
        self,
        path: str | Path,
        max_chars: int | None = 5_000,
        listeners: list[Listener] | None = None,
    ):
        """
        Args:
            path: The JSONL file to append to.
            max_chars: Longer text (tool outputs, thinking, replies) keeps its
                start and end, with a note saying how much was left out.
                None keeps everything.
            listeners: Called with every event, exactly as it was written.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars
        self.listeners = listeners or []

    def event(self, kind: str, **data) -> None:
        """
        Record an event to a JSONL file. Writes one line per event.

        Read it back with `pandas.read_json(path, lines=True)`.

        Args:
            kind: The type of event we are recording (e.g. 'tool', 'thinking')
        """
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            **self._shorten(data),  # type: ignore
        }

        line = json.dumps(record, default=_to_json)

        with self.path.open(mode="a", encoding="utf-8") as file:
            file.write(line + "\n")

        # Listeners get the event as it reads back from the file, so a live
        # view and a replay of the log (python -m wsp.watch) show the same thing.
        for listener in self.listeners:
            listener(json.loads(line))

    # -------------------------------------------------------------------------------- private

    def _shorten(self, value: object, key: str | None = None) -> object:
        """
        Shorten every long string in ``value``, looking inside dicts and lists.
        """
        if isinstance(value, dict):
            return {k: self._shorten(v, key=k) for k, v in value.items()}

        if isinstance(value, list):
            return [self._shorten(item, key=key) for item in value]

        if isinstance(value, str) and key not in KEEP_IN_FULL:
            return _truncate(value, self.max_chars)

        return value


def _truncate(text: str, max_chars: int | None) -> str:
    """
    Keep the first and last halves of ``max_chars``, e.g. a traceback's
    message or a reply's conclusion, and say how much was left out.
    """
    if max_chars is None or len(text) <= max_chars:
        return text

    half = max_chars // 2
    omitted = len(text) - 2 * half

    return f"{text[:half]}\n\n[... {omitted:,} characters omitted from the log ...]\n\n{text[-half:]}"


def _to_json(value: object) -> object:
    """
    Fallback for values `json` cannot serialise on its own.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    return repr(value)
