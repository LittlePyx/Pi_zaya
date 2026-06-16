from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException


def path_is_within(path_obj: Path, roots: list[Path]) -> bool:
    try:
        path = Path(path_obj).expanduser().resolve(strict=False)
    except Exception:
        return False
    for root in roots:
        try:
            root_resolved = Path(root).expanduser().resolve(strict=False)
            path.relative_to(root_resolved)
            return True
        except Exception:
            continue
    return False


def resolve_library_pdf_path_arg(path_raw: str, *, pdf_dir: Path) -> Path:
    raw = str(path_raw or "").strip()
    if not raw:
        raise HTTPException(400, "path required")
    pdf_d = Path(pdf_dir).expanduser().resolve(strict=False)
    try:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = pdf_d / candidate
        resolved = candidate.resolve(strict=False)
    except Exception as exc:
        raise HTTPException(400, f"invalid path: {exc}") from exc
    if resolved.suffix.lower() != ".pdf":
        raise HTTPException(400, "path must point to a PDF")
    if not path_is_within(resolved, [pdf_d]):
        raise HTTPException(400, "path must be within the configured PDF directory")
    return resolved


def resolve_library_pdf_name_arg(
    pdf_name: str,
    *,
    pdf_dir: Path,
    require_exists: bool = False,
    is_file: Callable[[Path], bool] | None = None,
) -> Path:
    raw = str(pdf_name or "").strip()
    if not raw:
        raise HTTPException(400, "invalid pdf_name")
    if Path(raw).name != raw or Path(raw.replace("\\", "/")).name != raw:
        raise HTTPException(400, "invalid pdf_name")
    resolved = resolve_library_pdf_path_arg(raw, pdf_dir=pdf_dir)
    if require_exists:
        exists = is_file(resolved) if is_file else resolved.is_file()
        if not exists:
            raise HTTPException(404, "pdf not found")
    return resolved
