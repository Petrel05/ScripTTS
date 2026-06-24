from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ScriptTask:
    id: str
    genre: str
    theme: str
    user_prompt: str
    constraints: dict[str, Any]


def read_jsonl(path: Path) -> list[ScriptTask]:
    tasks: list[ScriptTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            tasks.append(
                ScriptTask(
                    id=str(raw.get("id") or f"task_{line_no:03d}"),
                    genre=str(raw.get("genre") or ""),
                    theme=str(raw.get("theme") or ""),
                    user_prompt=str(raw.get("user_prompt") or raw.get("prompt") or ""),
                    constraints=dict(raw.get("constraints") or {}),
                )
            )
    return tasks


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
