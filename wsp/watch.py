"""
Watch a run: replay its log, or follow it live from another terminal.

    uv run python -m wsp.watch                         the latest run, replayed
    uv run python -m wsp.watch --follow                the latest run, live
    uv run python -m wsp.watch runs/mnist-zeros-... --view verbose
"""

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

from wsp.console import LEVELS, ConsoleView

POLL_SECONDS = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay or follow a run's log.")
    parser.add_argument("run", nargs="?", help="A run folder. Defaults to the latest in runs/.")
    parser.add_argument("--view", choices=list(LEVELS), default="normal")
    parser.add_argument("--follow", action="store_true", help="Keep showing events as they arrive.")
    parser.add_argument("--root", default="runs", help="Where to look for the latest run.")
    args = parser.parse_args()

    folder = Path(args.run) if args.run else latest_run(Path(args.root))
    view = ConsoleView(args.view)

    for event in read_events(folder / "log.jsonl", follow=args.follow):
        view(event)


def latest_run(root: Path) -> Path:
    runs = [path for path in root.iterdir() if (path / "log.jsonl").exists()]
    if not runs:
        raise SystemExit(f"No runs found in {root}/")
    return max(runs, key=lambda path: (path / "log.jsonl").stat().st_mtime)


def read_events(path: Path, follow: bool = False) -> Iterator[dict]:
    """
    Every event in the log. With ``follow``, wait for new ones until the run ends.
    """
    with path.open(encoding="utf-8") as file:
        while True:
            position = file.tell()
            line = file.readline()

            # A line still being written has no newline yet: wait for the rest.
            if not line.endswith("\n"):
                if not follow:
                    return
                file.seek(position)
                time.sleep(POLL_SECONDS)
                continue

            event = json.loads(line)
            yield event

            if follow and event.get("kind") == "end":
                return


if __name__ == "__main__":
    main()
