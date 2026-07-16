from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kb.converter.structured_indices import STRUCTURED_INDEX_VERSION, rebuild_structured_indices_for_markdown
from kb.source_filters import is_excluded_source_path

_DEFAULT_EXCLUDE_DIRS = {"temp", "__pycache__"}
_DEFAULT_EXCLUDE_NAMES = {"assets_manifest.md", "quality_report.md", "output.md"}
_STRUCTURED_INDEX_FILES = (
    "anchor_index.json",
    "equation_index.json",
    "figure_index.json",
    "reference_index.json",
    "table_index.json",
)


@dataclass
class StructuredIndexBatchStats:
    scanned: int = 0
    rebuilt: int = 0
    skipped: int = 0
    failed: int = 0
    citation_mention_count: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["version"] = int(STRUCTURED_INDEX_VERSION)
        return payload


def iter_markdown_files(
    src: Path | str,
    *,
    glob: str = "*.md",
    exclude_dirs: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> list[Path]:
    root = Path(src).expanduser()
    excluded_dirs = set(exclude_dirs or _DEFAULT_EXCLUDE_DIRS)
    excluded_names = set(exclude_names or _DEFAULT_EXCLUDE_NAMES)
    if root.is_file():
        return [root] if root.name not in excluded_names else []

    files: list[Path] = []
    for path in root.rglob(glob):
        if not path.is_file():
            continue
        if path.name in excluded_names:
            continue
        if is_excluded_source_path(str(path)):
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def structured_indices_need_rebuild(md_path: Path | str, *, assets_dir: Path | str | None = None) -> bool:
    md = Path(md_path).expanduser()
    assets = Path(assets_dir).expanduser() if assets_dir is not None else md.parent / "assets"
    try:
        md_mtime = float(md.stat().st_mtime)
    except Exception:
        md_mtime = 0.0

    for file_name in _STRUCTURED_INDEX_FILES:
        index_path = assets / file_name
        if not index_path.exists() or not index_path.is_file():
            return True
        payload = _load_json(index_path)
        try:
            version = int(payload.get("version") or 0)
        except Exception:
            version = 0
        if version < int(STRUCTURED_INDEX_VERSION):
            return True
        try:
            if md_mtime > 0 and float(index_path.stat().st_mtime) < md_mtime:
                return True
        except Exception:
            return True

    ref_payload = _load_json(assets / "reference_index.json")
    if "citation_mention_count" not in ref_payload:
        return True
    return False


def rebuild_structured_indices_for_root(
    src: Path | str,
    *,
    glob: str = "*.md",
    exclude_dirs: set[str] | None = None,
    exclude_names: set[str] | None = None,
    force: bool = False,
    max_errors: int = 20,
) -> dict[str, Any]:
    stats = StructuredIndexBatchStats()
    md_files = iter_markdown_files(
        src,
        glob=glob,
        exclude_dirs=exclude_dirs,
        exclude_names=exclude_names,
    )
    for md_path in md_files:
        stats.scanned += 1
        assets_dir = md_path.parent / "assets"
        if (not force) and (not structured_indices_need_rebuild(md_path, assets_dir=assets_dir)):
            stats.skipped += 1
            continue
        try:
            out = rebuild_structured_indices_for_markdown(md_path, assets_dir=assets_dir)
            ref_payload = out.get("reference_index") if isinstance(out, dict) else {}
            try:
                stats.citation_mention_count += int((ref_payload or {}).get("citation_mention_count") or 0)
            except Exception:
                pass
            stats.rebuilt += 1
        except Exception as exc:
            stats.failed += 1
            if len(stats.errors) < max(0, int(max_errors)):
                stats.errors.append({"path": str(md_path), "error": str(exc)})
    return stats.to_dict()
