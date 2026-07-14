from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
IMAGE_EXT_BY_PIL_FORMAT = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
}

ROOT_RELATIVE_FILE_ID_PREFIX = "kb-source/"


def resolved_path(value: Path | str | None) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve(strict=False)
    except Exception:
        return None


def unique_resolved_roots(values: list[Path | str | None]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for value in values:
        root = resolved_path(value)
        if root is None:
            continue
        key = str(root).casefold()
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def root_relative_file_id(
    path_obj: Path | str | None,
    roots: list[Path | str | None],
) -> str:
    path = resolved_path(path_obj)
    if path is None:
        return ""
    for index, root in enumerate(unique_resolved_roots(roots)):
        try:
            relative = path.relative_to(root)
        except Exception:
            continue
        parts = [part for part in relative.parts if part not in {"", "."}]
        if not parts or any(part == ".." for part in parts):
            return ""
        return f"{ROOT_RELATIVE_FILE_ID_PREFIX}{index}/{'/'.join(parts)}"
    return ""


def resolve_root_relative_file_id(
    value: Path | str | None,
    roots: list[Path | str | None],
) -> Path | None:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw.startswith(ROOT_RELATIVE_FILE_ID_PREFIX):
        return None
    tail = raw[len(ROOT_RELATIVE_FILE_ID_PREFIX) :]
    root_index_raw, separator, relative_raw = tail.partition("/")
    if not separator or not root_index_raw.isdigit():
        return None
    root_list = unique_resolved_roots(roots)
    root_index = int(root_index_raw)
    if root_index < 0 or root_index >= len(root_list):
        return None
    parts = relative_raw.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    root = root_list[root_index]
    return resolve_existing_file_under_roots(root.joinpath(*parts), [root])


def reference_source_roots(
    *,
    md_root: Path | str | None,
    db_dir: Path | str | None,
) -> list[Path]:
    db_path = resolved_path(db_dir)
    tmp_root = db_path.parent / "tmp" if db_path is not None else None
    return unique_resolved_roots([md_root, tmp_root])


def path_is_within_roots(path_obj: Path | str | None, roots: list[Path | str | None]) -> bool:
    path = resolved_path(path_obj)
    if path is None:
        return False
    for root in unique_resolved_roots(roots):
        try:
            path.relative_to(root)
            return True
        except Exception:
            continue
    return False


def resolve_existing_file_under_roots(
    path_obj: Path | str | None,
    roots: list[Path | str | None],
) -> Path | None:
    path = resolved_path(path_obj)
    if path is None or not path_is_within_roots(path, roots):
        return None
    try:
        if path.is_file():
            return path
    except Exception:
        return None
    return None


def _looks_like_url_suffix(suffix: str, *, separator: str) -> bool:
    tail = str(suffix or "").strip()
    if not tail or "/" in tail or "\\" in tail:
        return False
    low = tail.lower()
    if separator == "?":
        return "=" in tail or low in {"download", "reader", "viewer", "locate", "selection"}
    if low.startswith(("page=", "p=")):
        return True
    return "." not in tail and len(tail) <= 80


def _strip_source_path_url_suffix(raw: str) -> str:
    text = str(raw or "")
    if not text:
        return ""
    if text.lower().startswith("file://"):
        cut_at = len(text)
        for sep in ("?", "#"):
            idx = text.find(sep)
            if idx >= 0:
                cut_at = min(cut_at, idx)
        return text[:cut_at]
    cut_at = len(text)
    for sep in ("?", "#"):
        idx = text.find(sep)
        if idx >= 0 and _looks_like_url_suffix(text[idx + 1 :], separator=sep):
            cut_at = min(cut_at, idx)
    return text[:cut_at]


def clean_file_source_path_input(value: Path | str | None) -> str:
    raw = str(value or "").replace("\x00", " ").strip()
    if not raw:
        return ""
    raw = _strip_source_path_url_suffix(raw)
    lower = raw.lower()
    if lower.startswith("file:///"):
        raw = raw[8:]
    elif lower.startswith("file://"):
        raw = "//" + raw[7:]
    try:
        raw = unquote(raw)
    except Exception:
        pass
    return raw.strip()


def chat_image_upload_roots(db_dir: Path | str | None) -> list[Path]:
    db_root = resolved_path(db_dir)
    if db_root is None:
        return []
    return [db_root / "_chat_uploads" / "images"]


def _resolve_upload_leaf_under_root(path_obj: Path | str | None, root: Path | str | None) -> Path | None:
    root_path = resolved_path(root)
    if root_path is None:
        return None
    raw = clean_file_source_path_input(path_obj)
    if not raw:
        return None
    normalized = raw.replace("\\", "/")
    if "/" in normalized or ":" in normalized or normalized in {".", ".."}:
        return None
    return resolve_existing_file_under_roots(root_path / normalized, [root_path])


def resolve_chat_image_upload_path(path_obj: Path | str | None, *, db_dir: Path | str | None) -> Path | None:
    roots = chat_image_upload_roots(db_dir)
    direct = resolve_existing_file_under_roots(path_obj, roots)
    if direct is not None:
        return direct
    for root in roots:
        leaf = _resolve_upload_leaf_under_root(path_obj, root)
        if leaf is not None:
            return leaf
    return None


def sniff_image_ext(data: bytes | bytearray | memoryview | None) -> str:
    head = bytes(data or b"")[:32]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return ".gif"
    if (len(head) >= 12) and (head[:4] == b"RIFF") and (head[8:12] == b"WEBP"):
        return ".webp"
    if head.startswith(b"BM"):
        return ".bmp"
    return ""


def image_mime_for_ext(ext: str) -> str:
    return IMAGE_MIME_BY_EXT.get(str(ext or "").strip().lower(), "")


def image_ext_for_mime(mime: str) -> str:
    mime_norm = str(mime or "").strip().lower()
    for ext, ext_mime in IMAGE_MIME_BY_EXT.items():
        if ext_mime == mime_norm:
            return ext
    return ""


def sniff_image_mime(data: bytes | bytearray | memoryview | None) -> str:
    return image_mime_for_ext(sniff_image_ext(data))


def _pillow_verified_image_mime(source: Path | BytesIO) -> tuple[str, bool]:
    try:
        from PIL import Image
    except Exception:
        return "", False
    try:
        with Image.open(source) as img:
            fmt = str(getattr(img, "format", "") or "").strip().upper()
            width, height = getattr(img, "size", (0, 0))
            if int(width or 0) <= 0 or int(height or 0) <= 0:
                return "", True
            img.verify()
        ext = IMAGE_EXT_BY_PIL_FORMAT.get(fmt, "")
        return image_mime_for_ext(ext), True
    except Exception:
        return "", True


def verified_image_bytes_mime(data: bytes | bytearray | memoryview | None) -> str:
    raw = bytes(data or b"")
    if not raw:
        return ""
    mime, pillow_checked = _pillow_verified_image_mime(BytesIO(raw))
    if pillow_checked:
        return mime
    return sniff_image_mime(raw)


def verified_image_file_mime(path_obj: Path | str | None) -> str:
    path = resolved_path(path_obj)
    if path is None:
        return ""
    mime, pillow_checked = _pillow_verified_image_mime(path)
    if pillow_checked:
        return mime
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
    except Exception:
        return ""
    return sniff_image_mime(head)


def resolve_verified_chat_image_upload_path(
    path_obj: Path | str | None,
    *,
    db_dir: Path | str | None,
) -> tuple[Path, str] | None:
    return resolve_verified_image_file_under_roots(path_obj, chat_image_upload_roots(db_dir))


def resolve_verified_image_file_under_roots(
    path_obj: Path | str | None,
    roots: list[Path | str | None],
) -> tuple[Path, str] | None:
    path = resolve_existing_file_under_roots(path_obj, roots)
    if path is None:
        for root in unique_resolved_roots(roots):
            path = _resolve_upload_leaf_under_root(path_obj, root)
            if path is not None:
                break
    if path is None:
        return None
    mime = verified_image_file_mime(path)
    if not mime:
        return None
    return path, mime


def is_probably_pdf_bytes(data: bytes | bytearray | memoryview | None) -> bool:
    head = bytes(data or b"")[:1024]
    if head.startswith(b"\xef\xbb\xbf"):
        head = head[3:]
    return head.lstrip(b"\x00\t\n\r\f ").startswith(b"%PDF-")


def verified_pdf_file(path_obj: Path | str | None) -> bool:
    path = resolved_path(path_obj)
    if path is None or path.suffix.lower() != ".pdf":
        return False
    try:
        with path.open("rb") as fh:
            head = fh.read(1024)
    except Exception:
        return False
    return is_probably_pdf_bytes(head)


def resolve_verified_pdf_file_under_roots(
    path_obj: Path | str | None,
    roots: list[Path | str | None],
) -> Path | None:
    path = resolve_existing_file_under_roots(path_obj, roots)
    if path is None or not verified_pdf_file(path):
        return None
    return path
