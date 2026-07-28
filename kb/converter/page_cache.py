from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .config import canonical_speed_mode


PAGE_CACHE_SCHEMA_VERSION = 2
PAGE_OUTPUT_ALGORITHM_VERSION = 1
PAGE_CACHE_DIR_NAME = ".conversion_cache"

# Hash files whose whole purpose contributes to the cached, per-page Markdown
# or page assets.  Document assembly/finalization modules deliberately do not
# belong here: those stages run again after cached pages are loaded.
#
# ``pipeline.py`` currently mixes page-local helpers with document finalization,
# and ``post_processing.py`` contains one page-local fence repair alongside the
# document cleanup pipeline.  They are therefore governed by the explicit
# PAGE_OUTPUT_ALGORITHM_VERSION above instead of whole-file hashes.  Bump that
# version whenever a page-output helper in either mixed module changes.
_PAGE_OUTPUT_COMPONENTS = (
    "block_classifier.py",
    "config.py",
    "figure_assets.py",
    "formula_markdown.py",
    "geometry_utils.py",
    "heuristics.py",
    "layout_analysis.py",
    "llm_worker.py",
    "models.py",
    "page_figure_metadata.py",
    "page_image_markdown.py",
    "page_layout_crops.py",
    "page_local_pipeline.py",
    "page_table_fallback.py",
    "page_text_blocks.py",
    "page_vision_direct_page.py",
    "page_vision_guardrails.py",
    "pdf_reference_text.py",
    "pipeline_render_markdown.py",
    "pipeline_vision_direct.py",
    "post_heading_rules.py",
    "post_math_rules.py",
    "post_references.py",
    "reference_markdown.py",
    "reference_page_vl.py",
    "tables.py",
    "text_utils.py",
    "vision_circuit_breaker.py",
)
_QUALITY_ENV_KEYS = (
    "KB_LLM_HARD_TIMEOUT_S",
    "KB_PDF_COMPLEX_PAGE_FALLBACK",
    "KB_PDF_EQ_IMAGE_FALLBACK",
    "KB_PDF_FIGURE_DPI",
    "KB_PDF_LLM_VISION_MATH",
    "KB_PDF_SAFE_COMPLEX_FALLBACK",
    "KB_PDF_ULTRA_FAST_VISION_TIMEOUT_S",
    "KB_PDF_ULTRA_FAST_PAGE_BUDGET_S",
    "KB_PDF_VISION_COMPRESS",
    "KB_PDF_VISION_DPI",
    "KB_PDF_VISION_EMPTY_RETRY",
    "KB_PDF_VISION_FORMULA_DPI",
    "KB_PDF_VISION_FORMULA_MAX_PER_PAGE",
    "KB_PDF_VISION_FORMULA_OVERLAY",
    "KB_PDF_VISION_FRAGMENT_FALLBACK",
    "KB_PDF_VISION_HARD_TIMEOUT_S",
    "KB_PDF_VISION_LAYOUT_CROP_MODE",
    "KB_PDF_VISION_LAYOUT_CROP_MAX_TOKENS",
    "KB_PDF_VISION_MATH_POLICY",
    "KB_PDF_VISION_MATH_QUALITY_GATE",
    "KB_PDF_VISION_MAX_TOKENS",
    "KB_PDF_VISION_MIN_PX",
    "KB_PDF_VISION_PLAIN_PAGE_DPI",
    "KB_PDF_VISION_PLAIN_PAGE_MAX_TOKENS",
    "KB_PDF_VISION_PAGE_BUDGET_S",
    "KB_PDF_VISION_REFS_COLUMN_MODE",
    "KB_PDF_VISION_REFS_CROP_MAX_TOKENS",
    "KB_PDF_VISION_REFS_LOCAL_MIN_ENTRIES",
    "KB_PDF_VISION_REFS_MAX_TOKENS",
    "KB_PDF_VISION_REFS_PREFER_LOCAL",
    "KB_PDF_VISION_TIMEOUT_S",
)
_BAD_PAGE_OUTPUT_RE = re.compile(
    r"(?:\[page\s+\d+\s+conversion\s+incomplete\]|"
    r"conversion\s+incomplete|api\s+access\s+denied|"
    r"account\s+is\s+in\s+good\s+standing)",
    flags=re.I,
)
_ASSET_LINK_RE = re.compile(r"!\[[^\]]*\]\(\./assets/([^)]+)\)", flags=re.I)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(raw.encode("utf-8"))


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def page_markdown_is_reusable(markdown: str | None) -> bool:
    text = str(markdown or "").strip()
    if not text or _BAD_PAGE_OUTPUT_RE.search(text):
        return False
    visible = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return bool(visible.strip())


def _page_output_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    digest.update(f"page-output-algorithm:{PAGE_OUTPUT_ALGORITHM_VERSION}".encode("utf-8"))
    for name in _PAGE_OUTPUT_COMPONENTS:
        path = root / name
        digest.update(name.encode("utf-8"))
        try:
            digest.update(_sha256_file(path).encode("ascii"))
        except Exception:
            digest.update(b"missing")
    return digest.hexdigest()


def _config_payload(cfg: Any) -> dict[str, Any]:
    llm = getattr(cfg, "llm", None)
    endpoint = str(getattr(llm, "base_url", "") or "").strip()
    return {
        "translate_zh": bool(getattr(cfg, "translate_zh", False)),
        "llm_classify": bool(getattr(cfg, "llm_classify", True)),
        "llm_render_page": bool(getattr(cfg, "llm_render_page", False)),
        "llm_classify_only_if_needed": bool(getattr(cfg, "llm_classify_only_if_needed", True)),
        "dpi": int(getattr(cfg, "dpi", 200) or 200),
        "image_scale": float(getattr(cfg, "image_scale", 2.2) or 2.2),
        "figure_dpi": int(getattr(cfg, "figure_dpi", 0) or 0),
        "image_alpha": bool(getattr(cfg, "image_alpha", False)),
        "detect_tables": bool(getattr(cfg, "detect_tables", True)),
        "table_pdfplumber_fallback": bool(getattr(cfg, "table_pdfplumber_fallback", False)),
        "eq_image_fallback": bool(getattr(cfg, "eq_image_fallback", False)),
        "global_noise_scan": bool(getattr(cfg, "global_noise_scan", True)),
        "llm_repair": bool(getattr(cfg, "llm_repair", True)),
        "llm_repair_body_math": bool(getattr(cfg, "llm_repair_body_math", False)),
        "llm_smart_math_repair": bool(getattr(cfg, "llm_smart_math_repair", True)),
        "llm_render_max_tokens": int(getattr(cfg, "llm_render_max_tokens", 0) or 0),
        "speed_mode": canonical_speed_mode(getattr(cfg, "speed_mode", "normal")),
        "llm": {
            "model": str(getattr(llm, "model", "") or "").strip(),
            "temperature": float(getattr(llm, "temperature", 0.0) or 0.0),
            "max_tokens": int(getattr(llm, "max_tokens", 0) or 0),
            "endpoint_hash": _sha256_bytes(endpoint.encode("utf-8")) if endpoint else "",
        },
        "quality_environment": {
            key: str(os.environ.get(key, "") or "").strip()
            for key in _QUALITY_ENV_KEYS
        },
    }


def _legacy_config_payload(
    cfg: Any,
    *,
    classify_batch_size: int | None = None,
) -> dict[str, Any]:
    """Recreate the schema-v2 fingerprint used before batch size was ignored."""
    payload = _config_payload(cfg)
    batch_size = (
        getattr(cfg, "classify_batch_size", 40)
        if classify_batch_size is None
        else classify_batch_size
    )
    payload["classify_batch_size"] = int(batch_size or 40)
    return payload


class PageConversionCache:
    """Durable, per-page conversion cache with content and configuration validation."""

    def __init__(self, *, save_dir: Path, pdf_path: Path, cfg: Any, total_pages: int):
        self.save_dir = Path(save_dir)
        self.pdf_path = Path(pdf_path)
        self.total_pages = max(0, int(total_pages or 0))
        self.enabled = _env_enabled("KB_PDF_PAGE_CACHE", default=True)
        self.refresh = _env_enabled("KB_PDF_PAGE_CACHE_REFRESH", default=False)
        self.root = self.save_dir / PAGE_CACHE_DIR_NAME
        self.pages_dir = self.root / "pages"
        self._lock = threading.RLock()
        self.hits: set[int] = set()
        self.stores: set[int] = set()
        self.rejected: set[int] = set()
        self.targeted_retry_pages: set[int] = set()
        raw_retry_pages = str(os.environ.get("KB_PDF_PAGE_CACHE_RETRY_PAGES", "") or "").strip()
        for raw_page in re.split(r"[,;\s]+", raw_retry_pages):
            if not raw_page.isdigit():
                continue
            page_number = int(raw_page)
            if page_number > 0:
                self.targeted_retry_pages.add(page_number - 1)

        self.source_fingerprint = ""
        self.page_output_fingerprint = ""
        self.config_payload: dict[str, Any] = {}
        self.config_fingerprint = ""
        self.compatible_config_fingerprints: set[str] = set()
        if not self.enabled:
            return
        try:
            self.source_fingerprint = _sha256_file(self.pdf_path)
            self.page_output_fingerprint = _page_output_fingerprint()
            self.config_payload = _config_payload(cfg)
            self.config_fingerprint = _stable_json_hash({"config": self.config_payload})
            # Schema-v2 accidentally included a scheduling-only batch size in the
            # output fingerprint. Accept the two historical application defaults
            # so upgrading from CLI (40) to UI (80), or vice versa, keeps valid pages.
            legacy_batch_sizes = {
                int(getattr(cfg, "classify_batch_size", 40) or 40),
                40,
                80,
            }
            self.compatible_config_fingerprints = {self.config_fingerprint}
            self.compatible_config_fingerprints.update(
                _stable_json_hash(
                    {
                        "config": _legacy_config_payload(
                            cfg,
                            classify_batch_size=batch_size,
                        )
                    }
                )
                for batch_size in legacy_batch_sizes
            )
            self.pages_dir.mkdir(parents=True, exist_ok=True)
            self._write_manifest(status="active")
        except Exception:
            self.enabled = False

    def _page_fingerprint(self, page_index: int) -> str:
        return _stable_json_hash({
            "source_fingerprint": self.source_fingerprint,
            "page_index": int(page_index),
            "total_pages": self.total_pages,
        })

    def _page_dir(self, page_index: int) -> Path:
        return self.pages_dir / f"{int(page_index) + 1:05d}"

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _write_manifest(self, *, status: str) -> None:
        if not self.enabled:
            return
        payload = {
            "schema_version": PAGE_CACHE_SCHEMA_VERSION,
            "status": str(status or "active"),
            "source_name": self.pdf_path.name,
            "source_fingerprint": self.source_fingerprint,
            "page_output_algorithm_version": PAGE_OUTPUT_ALGORITHM_VERSION,
            "page_output_fingerprint": self.page_output_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "config": self.config_payload,
            "total_pages": self.total_pages,
            "hits": sorted(page + 1 for page in self.hits),
            "stored": sorted(page + 1 for page in self.stores),
            "rejected": sorted(page + 1 for page in self.rejected),
            "targeted_retry_pages": sorted(page + 1 for page in self.targeted_retry_pages),
            "updated_at": time.time(),
        }
        self._atomic_write(
            self.root / "manifest.json",
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    @staticmethod
    def _entry_assets(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows = entry.get("assets")
        return [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []

    def load_page(self, page_index: int, *, assets_dir: Path) -> str | None:
        if not self.enabled or self.refresh:
            return None
        if int(page_index) in self.targeted_retry_pages:
            with self._lock:
                self.rejected.add(int(page_index))
            return None
        page_dir = self._page_dir(page_index)
        entry_path = page_dir / "entry.json"
        if not entry_path.is_file():
            return None
        try:
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            if not isinstance(entry, dict):
                raise ValueError("invalid page cache entry")
            if int(entry.get("schema_version") or 0) != PAGE_CACHE_SCHEMA_VERSION:
                raise ValueError("page cache schema changed")
            if str(entry.get("page_fingerprint") or "") != self._page_fingerprint(page_index):
                raise ValueError("page source changed")
            if str(entry.get("page_output_fingerprint") or "") != self.page_output_fingerprint:
                raise ValueError("page output algorithm changed")
            entry_config_fingerprint = str(entry.get("config_fingerprint") or "")
            cross_profile_targeted_retry = bool(self.targeted_retry_pages)
            if entry_config_fingerprint not in self.compatible_config_fingerprints and not cross_profile_targeted_retry:
                raise ValueError("conversion configuration changed")
            # Keep cached text out of *.md discovery so ingestion cannot index
            # page fragments as standalone documents.
            markdown_bytes = (page_dir / "page.txt").read_bytes()
            if _sha256_bytes(markdown_bytes) != str(entry.get("markdown_sha256") or ""):
                raise ValueError("cached markdown checksum mismatch")
            markdown = markdown_bytes.decode("utf-8")
            if not page_markdown_is_reusable(markdown):
                raise ValueError("cached page is incomplete")

            cached_assets = self._entry_assets(entry)
            asset_names = {str(row.get("name") or "") for row in cached_assets}
            referenced = {str(name or "").strip() for name in _ASSET_LINK_RE.findall(markdown)}
            if any(name and name not in asset_names for name in referenced):
                raise ValueError("cached page asset is missing from manifest")
            for row in cached_assets:
                name = str(row.get("name") or "").strip()
                if not name or Path(name).name != name:
                    raise ValueError("invalid cached asset name")
                source = page_dir / "assets" / name
                if not source.is_file() or _sha256_file(source) != str(row.get("sha256") or ""):
                    raise ValueError("cached asset checksum mismatch")
            for row in cached_assets:
                name = str(row.get("name") or "").strip()
                source = page_dir / "assets" / name
                self._atomic_write(Path(assets_dir) / name, source.read_bytes())
            if (
                entry_config_fingerprint != self.config_fingerprint
                and entry_config_fingerprint in self.compatible_config_fingerprints
            ):
                entry["config_fingerprint"] = self.config_fingerprint
                self._atomic_write(
                    entry_path,
                    json.dumps(entry, ensure_ascii=False, indent=2).encode("utf-8"),
                )
            with self._lock:
                self.hits.add(int(page_index))
            return markdown
        except Exception:
            with self._lock:
                self.rejected.add(int(page_index))
            return None

    def _page_asset_paths(self, page_index: int, assets_dir: Path) -> list[Path]:
        page_no = int(page_index) + 1
        pattern = re.compile(rf"^page_{page_no}(?:_|\.).+", flags=re.I)
        return sorted(
            [path for path in Path(assets_dir).glob(f"page_{page_no}*") if path.is_file() and pattern.match(path.name)],
            key=lambda path: path.name.lower(),
        )

    def store_page(self, page_index: int, markdown: str | None, *, assets_dir: Path) -> bool:
        if not self.enabled or not page_markdown_is_reusable(markdown):
            return False
        text = str(markdown or "").strip()
        page_dir = self._page_dir(page_index)
        cached_assets_dir = page_dir / "assets"
        try:
            asset_rows: list[dict[str, Any]] = []
            for source in self._page_asset_paths(page_index, Path(assets_dir)):
                data = source.read_bytes()
                self._atomic_write(cached_assets_dir / source.name, data)
                asset_rows.append({
                    "name": source.name,
                    "size": len(data),
                    "sha256": _sha256_bytes(data),
                })
            referenced = {str(name or "").strip() for name in _ASSET_LINK_RE.findall(text)}
            if any(name and name not in {row["name"] for row in asset_rows} for name in referenced):
                return False
            markdown_bytes = text.encode("utf-8")
            self._atomic_write(page_dir / "page.txt", markdown_bytes)
            entry = {
                "schema_version": PAGE_CACHE_SCHEMA_VERSION,
                "status": "ready",
                "page_index": int(page_index),
                "page_number": int(page_index) + 1,
                "page_fingerprint": self._page_fingerprint(page_index),
                "page_output_fingerprint": self.page_output_fingerprint,
                "config_fingerprint": self.config_fingerprint,
                "markdown_sha256": _sha256_bytes(markdown_bytes),
                "markdown_chars": len(text),
                "assets": asset_rows,
                "created_at": time.time(),
            }
            self._atomic_write(
                page_dir / "entry.json",
                json.dumps(entry, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            with self._lock:
                self.stores.add(int(page_index))
            return True
        except Exception:
            return False

    def finish(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            try:
                self._write_manifest(status="ready")
            except Exception:
                pass
