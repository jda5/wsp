"""
Watch a run: follow it live from another terminal, or replay it afterwards.

    uv run python -m wsp.watch --follow                the run in progress, or the next to start
    uv run python -m wsp.watch                         the latest run, replayed
    uv run python -m wsp.watch runs/mnist-zeros-... --view verbose

Runs only write their log (log.jsonl); this is the one place it is displayed.
"""

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

from wsp.console import LEVELS, ConsoleView

POLL_SECONDS = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Follow or replay a run's log.")
    parser.add_argument(
        "run", nargs="?", help="A run folder. Defaults to the latest in --root."
    )
    parser.add_argument("--view", choices=list(LEVELS), default="normal")
    parser.add_argument(
        "--follow", action="store_true", help="Show events as they arrive."
    )
    parser.add_argument("--root", default="runs", help="Where runs are saved.")
    args = parser.parse_args()

    root = Path(args.root)

    if args.run:
        folder = Path(args.run)
    elif args.follow:
        folder = run_to_follow(root)
    else:
        folder = latest_run(root)
        if folder is None:
            raise SystemExit(f"No runs found in {root}/")

    view = ConsoleView(args.view)

    try:
        for event in read_events(folder / "log.jsonl", follow=args.follow):
            view(event)
    except KeyboardInterrupt:
        pass  # Ctrl-C stops watching; the run itself carries on


# -------------------------------------------------------------------------------- finding the run


def run_to_follow(root: Path) -> Path:
    """
    The run in progress, or, if none is, the next one to start.
    """
    latest = latest_run(root)
    if latest is not None and not _has_ended(latest):
        return latest

    print(f"Waiting for a run to start in {root}/ ...")
    known = set(_runs(root))

    while True:
        new = [folder for folder in _runs(root) if folder not in known]
        if new:
            return max(new, key=_last_written)
        time.sleep(POLL_SECONDS)


def latest_run(root: Path) -> Path | None:
    runs = _runs(root)
    return max(runs, key=_last_written) if runs else None


def _runs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [folder for folder in root.iterdir() if (folder / "log.jsonl").exists()]


def _last_written(folder: Path) -> float:
    return (folder / "log.jsonl").stat().st_mtime


def _has_ended(folder: Path) -> bool:
    lines = (folder / "log.jsonl").read_text(encoding="utf-8").splitlines()
    return bool(lines) and json.loads(lines[-1]).get("kind") == "end"


# -------------------------------------------------------------------------------- reading the log


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
