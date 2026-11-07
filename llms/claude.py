"""
A small, configurable Claude client for weight-space-programming experiments.

The point of this wrapper is to make it easy to *constrain* the model and compare
how it behaves: switch server tools (web search / web fetch / code execution) on or
off, turn thinking off where the model allows it, swap the base model, attach
files, and pull back any files the model produced in its code-execution sandbox
(e.g. a `weights.npz`).

Quick start:

    from llms.claude import Claude, ClaudeConfig

    claude = Claude(ClaudeConfig(code_execution=True, web_search=False))
    result = claude.ask("Write MNIST MLP weights to weights.npz without training.")
    print(result.text)
    print(result.files)          # local paths of downloaded sandbox files
    claude.save_transcript("runs/mnist-opus.json")

Auth: `ANTHROPIC_API_KEY` (a `.env` file in the project root is loaded), or an
`ant auth login` profile.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import anthropic
from dotenv import load_dotenv

load_dotenv()

Effort = Literal["low", "medium", "high", "xhigh", "max"]
Thinking = Literal["adaptive", "off"]

DEFAULT_MODEL = "claude-opus-5-5"

# Models that cannot run with thinking switched off at all (effort is the only lever).
_THINKING_ALWAYS_ON = ("claude-opus-5-5", "claude-fable-5", "claude-mythos-5")

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
)

_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".py",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
}


@dataclass
class ClaudeConfig:
    model: str = DEFAULT_MODEL
    system: str | None = None
    max_tokens: int = 64_000

    # ---------------------------------------- reasoning

    thinking: Thinking = "adaptive"

    # Opus 5.5 / Haiku 5.5 default to "medium"; set explicitly so runs are comparable.
    effort: Effort | None = "high"

    # "summarized" returns a readable summary of the reasoning; "omitted" returns
    # empty thinking blocks (thinking still happens and is billed either way).
    thinking_display: Literal["summarized", "omitted"] = "summarized"

    # ---------------------------------------- tools

    web_search: bool = False
    web_fetch: bool = False
    code_execution: bool = False
    # Only files at the top level of $OUTPUT_DIR come back from the sandbox; files
    # written anywhere else stay there. When code execution is on, this note is
    # added to the system prompt so the model saves its deliverables in the right
    # place. Turn it off if you want the system prompt to be exactly `system`.
    output_dir_hint: bool = True
    max_web_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    # Raw tool dicts appended as-is (e.g. a custom tool you handle yourself).
    extra_tools: list[dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------- misc

    # Server-side refusal fallback. Off by default here: if it fires, a different
    # model answers, which would confound model-vs-model comparisons. When on,
    # `Result.fallback_used` tells you whether it happened.
    fallbacks: bool = False
    # How many times to resume a server-tool turn that stops with `pause_turn`.
    max_continuations: int = 10
    output_dir: Path = Path("outputs")
    timeout: float = 30 * 60


@dataclass
class Result:
    text: str
    thinking: list[str]
    tool_calls: list[dict[str, Any]]
    files: list[Path]
    stop_reason: str | None
    model: str
    fallback_used: bool
    usage: dict[str, int]

    def __str__(self) -> str:
        return self.text


class Claude:
    """
    Stateful conversation with Claude. Each `ask()` appends to the history.
    """

    def __init__(
        self,
        config: ClaudeConfig | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        self.config = config or ClaudeConfig()
        self.client = client or anthropic.Anthropic(timeout=self.config.timeout)
        self.messages: list[dict[str, Any]] = []
        self.container_id: str | None = (
            None  # reused so sandbox files persist across turns
        )
        self._validate()

    # -------------------------------------------------------------------------------- public

    def ask(self, prompt: str, attachments: list[str | Path] | None = None) -> Result:
        content = self._attachment_blocks(attachments or []) + [
            {"type": "text", "text": prompt}
        ]
        self.messages.append({"role": "user", "content": content})

        usage: dict[str, int] = {}
        message = None
        for _ in range(self.config.max_continuations + 1):
            message = self._call()
            # Append the full content (not just text): thinking blocks, tool blocks
            # and compaction state must be replayed unchanged.
            self.messages.append({"role": "assistant", "content": message.content})
            _add_usage(usage, message.usage)
            if message.container is not None:
                self.container_id = message.container.id
            if message.stop_reason != "pause_turn":
                break
        assert message is not None

        return Result(
            text="".join(b.text for b in message.content if b.type == "text"),
            thinking=[
                b.thinking
                for m in self._turn_messages()
                for b in m
                if b.type == "thinking" and b.thinking
            ],
            tool_calls=[
                {"name": b.name, "input": b.input}
                for m in self._turn_messages()
                for b in m
                if b.type in ("server_tool_use", "tool_use")
            ],
            files=self._download_files(),
            stop_reason=message.stop_reason,
            model=message.model,
            fallback_used=any(
                b.type == "fallback" for m in self._turn_messages() for b in m
            ),
            usage=usage,
        )

    def reset(self) -> None:
        self.messages.clear()
        self.container_id = None

    def save_transcript(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(self.config).items()
            },
            "messages": [
                {
                    "role": m["role"],
                    "content": [
                        b.model_dump(mode="json") if hasattr(b, "model_dump") else b
                        for b in m["content"]
                    ],
                }
                for m in self.messages
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    # -------------------------------------------------------------------------------- private

    def _call(self):
        cfg = self.config
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "messages": self.messages,
        }
        betas: list[str] = []
        system = cfg.system or ""
        if cfg.code_execution and cfg.output_dir_hint:
            system += (
                "\n\nTo hand a file back to the user, save or copy it to the top level of "
                '"$OUTPUT_DIR" in the same command that creates it. Files elsewhere are not returned.'
            )
        if system.strip():
            kwargs["system"] = system.strip()
        if (thinking := self._thinking_param()) is not None:
            kwargs["thinking"] = thinking
        if cfg.effort:
            kwargs["output_config"] = {"effort": cfg.effort}
        if tools := self._tools():
            kwargs["tools"] = tools
        if self.container_id and cfg.code_execution:
            kwargs["container"] = self.container_id
        if cfg.fallbacks and cfg.model.startswith(_FALLBACK_MODELS):
            kwargs["fallbacks"] = "default"
            betas.append("server-side-fallback-2026-07-01")
        if betas:
            kwargs["betas"] = betas

        # Streaming avoids HTTP timeouts on long, high-max_tokens turns.
        with self.client.beta.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    def _thinking_param(self) -> dict[str, Any] | None:
        cfg = self.config
        if cfg.thinking == "adaptive":
            return {"type": "adaptive", "display": cfg.thinking_display}
        if cfg.model.startswith("claude-sonnet-5-5"):
            return {"type": "between_tools"}
        return {"type": "disabled"}

    def _tools(self) -> list[dict[str, Any]]:
        cfg = self.config
        # Haiku models only take the basic web tool variants.
        dynamic = "haiku" not in cfg.model
        web_opts: dict[str, Any] = {}
        if cfg.max_web_uses is not None:
            web_opts["max_uses"] = cfg.max_web_uses
        if cfg.allowed_domains:
            web_opts["allowed_domains"] = cfg.allowed_domains
        if cfg.blocked_domains:
            web_opts["blocked_domains"] = cfg.blocked_domains

        tools: list[dict[str, Any]] = []
        if cfg.web_search:
            t = "web_search_20260209" if dynamic else "web_search_20250305"
            tools.append({"type": t, "name": "web_search", **web_opts})
        if cfg.web_fetch:
            t = "web_fetch_20260209" if dynamic else "web_fetch_20250910"
            tools.append({"type": t, "name": "web_fetch", **web_opts})
        if cfg.code_execution:
            tools.append({"type": "code_execution_20260521", "name": "code_execution"})
        return tools + list(cfg.extra_tools)

    def _validate(self) -> None:
        cfg = self.config
        if cfg.thinking == "off":
            if cfg.model.startswith(_THINKING_ALWAYS_ON):
                raise ValueError(
                    f"{cfg.model} cannot run with thinking off. Use effort='low' to minimise "
                    "reasoning, or pick a model that allows it (e.g. claude-opus-5, "
                    "claude-sonnet-5-5, claude-opus-4-8)."
                )
            if cfg.effort in ("xhigh", "max") and cfg.model.startswith(
                ("claude-sonnet-5-5", "claude-opus-5", "claude-haiku-5-5")
            ):
                raise ValueError(
                    f"{cfg.model} only allows thinking off at effort 'high' or below."
                )

    def _attachment_blocks(self, paths: list[str | Path]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for p in map(Path, paths):
            if self.config.code_execution:
                # Put the file in the sandbox so the model can load it from code
                # (works for any type: .npy, .npz, .csv, ...).
                uploaded = self.client.files.upload(file=p)
                blocks.append({"type": "container_upload", "file_id": uploaded.id})
                continue

            mime = mimetypes.guess_type(p.name)[0] or ""
            if mime in _IMAGE_TYPES:
                data = base64.standard_b64encode(p.read_bytes()).decode()
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": data},
                    }
                )
            elif mime == "application/pdf":
                data = base64.standard_b64encode(p.read_bytes()).decode()
                blocks.append(
                    {
                        "type": "document",
                        "title": p.name,
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": data,
                        },
                    }
                )
            elif p.suffix.lower() in _TEXT_SUFFIXES:
                blocks.append(
                    {
                        "type": "document",
                        "title": p.name,
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": p.read_text(),
                        },
                    }
                )
            else:
                raise ValueError(
                    f"Can't inline {p.name}; enable code_execution to attach binary files."
                )
        return blocks

    def _download_files(self) -> list[Path]:
        """
        Download every file the sandbox produced during the latest turn.
        """
        file_ids: list[str] = []
        for content in self._turn_messages():
            for block in content:
                if block.type.endswith("_tool_result"):
                    _collect_file_ids(block.model_dump(), file_ids)

        saved: list[Path] = []
        if not file_ids:
            return saved
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for file_id in dict.fromkeys(file_ids):
            meta = self.client.files.retrieve_metadata(file_id)
            name = os.path.basename(
                meta.filename
            )  # never trust a path from the sandbox
            if not name or name in (".", ".."):
                continue
            dest = self.config.output_dir / name
            self.client.files.download(file_id).write_to_file(dest)
            saved.append(dest)
        return saved

    def _turn_messages(self) -> list[list[Any]]:
        """
        Assistant content lists produced since the last user message.
        """
        out: list[list[Any]] = []
        for m in reversed(self.messages):
            if m["role"] == "user":
                break
            out.append(m["content"])
        return list(reversed(out))


def _collect_file_ids(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("file_id"), str):
            out.append(node["file_id"])
        for v in node.values():
            _collect_file_ids(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_file_ids(v, out)


def _add_usage(total: dict[str, int], usage: Any) -> None:
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        total[key] = total.get(key, 0) + (getattr(usage, key, None) or 0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ask Claude something with configurable constraints."
    )
    ap.add_argument("prompt")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--effort", choices=["low", "medium", "high", "xhigh", "max"], default="high"
    )
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--web-search", action="store_true")
    ap.add_argument("--web-fetch", action="store_true")
    ap.add_argument(
        "--code", action="store_true", help="enable the code execution sandbox"
    )
    ap.add_argument("--attach", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--transcript", type=Path)
    args = ap.parse_args()

    claude = Claude(
        ClaudeConfig(
            model=args.model,
            effort=args.effort,
            thinking="off" if args.no_thinking else "adaptive",
            web_search=args.web_search,
            web_fetch=args.web_fetch,
            code_execution=args.code,
            output_dir=args.out,
        )
    )
    result = claude.ask(args.prompt, attachments=args.attach)
    for t in result.thinking:
        print(f"[thinking] {t}\n")
    for call in result.tool_calls:
        print(f"[tool] {call['name']}: {json.dumps(call['input'])[:200]}")
    print(result.text)
    if result.files:
        print("\nFiles:", *result.files, sep="\n  ")
    print(f"\n[{result.model} | stop={result.stop_reason} | usage={result.usage}]")
    if args.transcript:
        claude.save_transcript(args.transcript)


if __name__ == "__main__":
    main()
