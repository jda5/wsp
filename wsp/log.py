import json
from datetime import datetime
from pathlib import Path

import numpy as np


class RunLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **data) -> None:
        """
        Record an event to a JSONL file. Writes one line per event.

        Read it back with `pandas.read_json(path, lines=True)`.

        Args:
            kind: The type of event we are recording (e.g. 'tool', )
        """
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            **data,
        }

        with self.path.open(mode="a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=_to_json) + "\n")


def _to_json(value: object) -> object:
    """
    Fallback for values `json` cannot serialise on its own.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    return repr(value)
