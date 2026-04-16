from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_paper_guide_index_payload(md_path: Path | str, file_name: str) -> dict[str, Any]:
    path = Path(str(md_path or "")).expanduser()
    index_path = path.parent / "assets" / str(file_name or "").strip()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_paper_guide_index_rows(md_path: Path | str, *, file_name: str, key: str) -> list[dict]:
    payload = _load_paper_guide_index_payload(md_path, file_name)
    rows = payload.get(key)
    if not isinstance(rows, list):
        return []
    return [dict(item) for item in rows if isinstance(item, dict)]


def load_paper_guide_anchor_index(md_path: Path | str) -> list[dict]:
    return _load_paper_guide_index_rows(md_path, file_name="anchor_index.json", key="anchors")


def load_paper_guide_equation_index(md_path: Path | str) -> list[dict]:
    return _load_paper_guide_index_rows(md_path, file_name="equation_index.json", key="equations")


def load_paper_guide_figure_index(md_path: Path | str) -> list[dict]:
    return _load_paper_guide_index_rows(md_path, file_name="figure_index.json", key="figures")


def load_paper_guide_reference_index(md_path: Path | str) -> list[dict]:
    return _load_paper_guide_index_rows(md_path, file_name="reference_index.json", key="references")
