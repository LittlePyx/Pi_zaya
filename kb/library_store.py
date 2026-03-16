# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import replace
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from kb.file_ops import _resolve_md_output_paths, _sha1_bytes
from kb.prefs import load_prefs as _load_local_prefs


_READING_STATUS_SET = {"", "unread", "reading", "done", "revisit"}

_CATEGORY_SUGGESTION_RULES: dict[str, tuple[str, ...]] = {
    "NeRF": ("nerf", "radiance field", "neural radiance", "novel view synthesis"),
    "3DGS": ("3dgs", "gaussian splatting", "gaussian splat", "3d gaussian"),
    "SCI": ("snapshot compressive", "compressive image", "compressive sensing", "high speed imaging", "high-speed imaging"),
    "Single-Photon Imaging": ("single photon imaging", "single-photon imaging", "single photon", "photon counting"),
    "Single-Pixel Imaging": ("single pixel imaging", "single-pixel imaging", "single pixel", "bucket detector"),
    "Inverse Imaging": ("inverse imaging", "inverse problem", "inverse problems"),
    "Diffusion": ("diffusion", "latent diffusion", "denoising diffusion", "score distillation"),
}

_TAG_SUGGESTION_RULES: dict[str, tuple[str, ...]] = {
    "single-image": ("single image", "single-image", "single snapshot"),
    "pose-free": ("pose free", "pose-free", "camera pose free", "without camera pose"),
    "real-time": ("real time", "real-time"),
    "novel-view-synthesis": ("novel view synthesis", "view synthesis"),
    "camera-pose": ("camera pose", "camera poses", "pose estimation"),
    "view-consistency": ("view consistency", "cross view consistency", "multi view consistency"),
    "gaussian-splatting": ("gaussian splatting", "gaussian splat", "3dgs"),
    "compressive-sensing": ("compressive sensing", "snapshot compressive", "compressive image"),
    "high-speed-imaging": ("high speed imaging", "high-speed imaging"),
    "physics-informed": ("physics informed", "physics-informed", "physics guided", "physics-guided", "physical prior"),
    "single-photon": ("single photon", "single-photon", "photon counting"),
    "single-pixel": ("single pixel", "single-pixel", "bucket detector"),
    "inverse-imaging": ("inverse imaging", "inverse problem", "inverse problems"),
    "image-reconstruction": ("image reconstruction", "reconstruct image", "reconstructed image"),
    "high-resolution": ("high resolution", "high-resolution", "super resolution", "super-resolution"),
    "low-light": ("low light", "low-light", "photon limited", "photon-limited"),
}

_SUGGESTION_MD_MAX_CHARS = 240_000
_SUGGESTION_SIGNAL_TEXT_MAX_CHARS = 3_600
_MD_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$")
_MD_KEYWORDS_RE = re.compile(r"(?im)^\s*(?:\*\*)?(?:keywords?|index terms?)(?:\*\*)?\s*[:\uFF1A-]\s*(.+)$")
_MD_CODE_FENCE_RE = re.compile(r"(?is)```.*?```")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_MD_DISPLAY_MATH_RE = re.compile(r"(?is)\$\$.*?\$\$|\\begin\{[^}]+\}.*?\\end\{[^}]+\}")
_MD_INLINE_MATH_RE = re.compile(r"\$[^$\n]{1,240}\$")
_MD_SECTION_ABSTRACT_HINTS = ("abstract", "summary")
_MD_SECTION_INTRO_HINTS = ("introduction", "intro", "background")
_MD_SECTION_METHOD_HINTS = ("method", "approach", "framework", "formulation", "methodology")
_MD_SECTION_CONTRIB_HINTS = ("contribution", "our contribution", "main contribution")
_MD_SECTION_CONCLUSION_HINTS = ("conclusion", "discussion", "limitation", "future work")
_LLM_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_GENERIC_DOC_TYPE_CATEGORY_RULES: dict[str, tuple[str, ...]] = {
    "Survey": ("survey", "review", "overview", "taxonomy"),
    "Dataset": ("dataset",),
    "Benchmark": ("benchmark",),
}
_GENERIC_DOC_TYPE_CATEGORY_LABELS = set(_GENERIC_DOC_TYPE_CATEGORY_RULES.keys())
_GENERIC_DOC_TYPE_TAGS = {"survey", "dataset", "benchmark"}
_GENERIC_DOC_TYPE_SIGNAL_KINDS = {"title", "display_title", "md_title", "summary", "keywords", "md_keywords", "headings"}
_SYSTEM_CATEGORY_HINTS: dict[str, str] = {
    "NeRF": "radiance-field and novel-view-synthesis papers",
    "3DGS": "3D Gaussian Splatting papers",
    "SCI": "snapshot compressive imaging papers",
    "Single-Photon Imaging": "single-photon or photon-counting imaging papers",
    "Single-Pixel Imaging": "single-pixel or bucket-detector imaging papers",
    "Inverse Imaging": "inverse-problem / inverse-imaging papers",
    "Diffusion": "diffusion-model papers",
    "Survey": "survey, review, overview, or taxonomy papers",
    "Dataset": "papers whose primary contribution is a dataset",
    "Benchmark": "papers whose primary contribution is a benchmark",
}
_SYSTEM_TAG_HINTS: dict[str, str] = {
    "single-image": "single-image or single-shot setting",
    "pose-free": "no camera pose required",
    "real-time": "real-time or interactive runtime",
    "novel-view-synthesis": "novel-view-synthesis task",
    "camera-pose": "camera pose estimation or dependence",
    "view-consistency": "cross-view or multi-view consistency",
    "gaussian-splatting": "gaussian-splatting based method",
    "compressive-sensing": "compressive sensing or snapshot compressive imaging",
    "high-speed-imaging": "high-speed imaging setting",
    "physics-informed": "uses physics priors or forward-model constraints",
    "single-photon": "single-photon or photon-counting acquisition",
    "single-pixel": "single-pixel / bucket-detector acquisition",
    "inverse-imaging": "inverse-imaging or inverse-problem formulation",
    "image-reconstruction": "image reconstruction task",
    "high-resolution": "high-resolution or super-resolution target",
    "low-light": "low-light or photon-limited setting",
}
_BAD_NEW_CATEGORY_NORMS = {
    "paper",
    "papers",
    "method",
    "methods",
    "model",
    "models",
    "approach",
    "framework",
    "deep learning",
    "machine learning",
    "computer vision",
    "imaging",
}
_BAD_NEW_TAG_NORMS = {
    "paper",
    "papers",
    "study",
    "method",
    "methods",
    "model",
    "models",
    "approach",
    "framework",
    "network",
    "networks",
    "deep learning",
    "machine learning",
    "imaging",
    "dataset",
    "datasets",
    "benchmark",
    "survey",
    "review",
}


class LibraryStore:
    """
    Minimal PDF library index:
    - keyed by sha1 to detect duplicates quickly
    - stores final pdf path and created_at
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pdf_files (
                  sha1 TEXT PRIMARY KEY,
                  path TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  citation_meta TEXT
                );
                """
            )
            # Add citation_meta column if it doesn't exist (for existing databases)
            try:
                conn.execute("ALTER TABLE pdf_files ADD COLUMN citation_meta TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_meta (
                  sha1 TEXT PRIMARY KEY,
                  paper_category TEXT NOT NULL DEFAULT '',
                  reading_status TEXT NOT NULL DEFAULT '',
                  note TEXT NOT NULL DEFAULT '',
                  suggested_category TEXT NOT NULL DEFAULT '',
                  dismissed_category TEXT NOT NULL DEFAULT '',
                  suggestion_updated_at REAL NOT NULL DEFAULT 0,
                  updated_at REAL NOT NULL DEFAULT 0,
                  FOREIGN KEY (sha1) REFERENCES pdf_files(sha1) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_tags (
                  sha1 TEXT NOT NULL,
                  tag_norm TEXT NOT NULL,
                  tag_label TEXT NOT NULL,
                  source TEXT NOT NULL DEFAULT 'user',
                  updated_at REAL NOT NULL DEFAULT 0,
                  PRIMARY KEY (sha1, tag_norm, source),
                  FOREIGN KEY (sha1) REFERENCES pdf_files(sha1) ON DELETE CASCADE
                );
                """
            )
            for ddl in (
                "ALTER TABLE paper_meta ADD COLUMN suggested_category TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE paper_meta ADD COLUMN dismissed_category TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE paper_meta ADD COLUMN suggestion_updated_at REAL NOT NULL DEFAULT 0",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass

    def _normalize_suggestion_text(self, value: str | None) -> str:
        text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
        return " ".join(text.split())

    def _text_contains_phrase(self, text_norm: str, phrase: str) -> bool:
        phrase_norm = self._normalize_suggestion_text(phrase)
        if not phrase_norm:
            return False
        return f" {phrase_norm} " in f" {text_norm} "

    def _decode_citation_meta(self, raw_meta: str | None) -> dict:
        if not raw_meta:
            return {}
        try:
            parsed = json.loads(raw_meta)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _row_value(self, row: sqlite3.Row | dict, key: str) -> object:
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            return None

    def _limit_suggestion_text(self, value: str | None, *, limit: int = _SUGGESTION_SIGNAL_TEXT_MAX_CHARS) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text[: max(1, int(limit))]

    def _iter_nested_text(self, value: object, *, max_items: int = 12, max_depth: int = 2) -> list[str]:
        out: list[str] = []

        def _walk(cur: object, depth: int) -> None:
            if len(out) >= max_items or depth < 0 or cur is None:
                return
            if isinstance(cur, str):
                clean = self._limit_suggestion_text(cur, limit=600)
                if clean:
                    out.append(clean)
                return
            if isinstance(cur, (list, tuple, set)):
                for item in cur:
                    _walk(item, depth - 1)
                    if len(out) >= max_items:
                        return
                return
            if isinstance(cur, dict):
                preferred_keys = (
                    "label",
                    "display_name",
                    "name",
                    "title",
                    "keyword",
                    "keywords",
                    "subject",
                    "subjects",
                    "topic",
                    "topics",
                    "summary",
                    "summary_line",
                    "description",
                    "text",
                )
                for key in preferred_keys:
                    if key not in cur:
                        continue
                    _walk(cur.get(key), depth - 1)
                    if len(out) >= max_items:
                        return
                for item in cur.values():
                    if isinstance(item, (str, list, tuple, set, dict)):
                        _walk(item, depth - 1)
                        if len(out) >= max_items:
                            return

        _walk(value, max_depth)
        return out[:max_items]

    def _strip_markdown_for_suggestions(self, raw_text: str | None) -> str:
        text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not text:
            return ""
        text = _MD_CODE_FENCE_RE.sub(" ", text)
        text = _MD_DISPLAY_MATH_RE.sub(" ", text)
        text = _MD_INLINE_MATH_RE.sub(" ", text)
        text = _MD_IMAGE_RE.sub(" ", text)
        text = _MD_LINK_RE.sub(r" \1 ", text)
        text = _MD_INLINE_CODE_RE.sub(r" \1 ", text)
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
        text = re.sub(r"(?m)^\s*>\s*", "", text)
        text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
        text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
        text = re.sub(r"(?m)^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*$", " ", text)
        text = re.sub(r"\|", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _markdown_heading_matches(self, heading: str, hints: Iterable[str]) -> bool:
        heading_norm = self._normalize_suggestion_text(heading)
        if not heading_norm:
            return False
        return any(self._text_contains_phrase(heading_norm, hint) for hint in hints)

    def _extract_markdown_signal_parts(self, md_path: Path | None) -> dict[str, str]:
        if md_path is None:
            return {}
        p = Path(md_path)
        try:
            if (not p.exists()) or (not p.is_file()):
                return {}
            raw_text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {}
        if not raw_text:
            return {}
        raw_text = raw_text[:_SUGGESTION_MD_MAX_CHARS]
        heading_matches = list(_MD_HEADING_RE.finditer(raw_text))
        headings = [self._strip_markdown_for_suggestions(match.group(1)) for match in heading_matches[:18]]
        sections: list[tuple[str, str]] = []
        for idx, match in enumerate(heading_matches[:36]):
            heading = self._strip_markdown_for_suggestions(match.group(1))
            if not heading:
                continue
            start = match.end()
            end = heading_matches[idx + 1].start() if (idx + 1) < len(heading_matches) else len(raw_text)
            body = self._strip_markdown_for_suggestions(raw_text[start:end])
            if body:
                sections.append((heading, body))

        title = headings[0] if headings else ""
        after_title = raw_text[heading_matches[0].end() :] if heading_matches else raw_text
        lead_paragraphs: list[str] = []
        for block in re.split(r"\n\s*\n", after_title):
            clean_block = self._strip_markdown_for_suggestions(block)
            if clean_block:
                lead_paragraphs.append(clean_block)
        keywords: list[str] = []
        for match in _MD_KEYWORDS_RE.finditer(raw_text):
            clean_keyword = self._strip_markdown_for_suggestions(match.group(1))
            if clean_keyword:
                keywords.append(clean_keyword)

        def _pick_section(hints: Iterable[str]) -> str:
            for heading, body in sections:
                if self._markdown_heading_matches(heading, hints):
                    return self._limit_suggestion_text(body, limit=2200)
            return ""

        abstract_text = _pick_section(_MD_SECTION_ABSTRACT_HINTS)
        intro_text = _pick_section(_MD_SECTION_INTRO_HINTS)
        method_text = _pick_section(_MD_SECTION_METHOD_HINTS)
        contribution_text = _pick_section(_MD_SECTION_CONTRIB_HINTS)
        conclusion_text = _pick_section(_MD_SECTION_CONCLUSION_HINTS)
        lead_text = self._limit_suggestion_text(" ".join(lead_paragraphs[:3]), limit=2200)
        return {
            "title": self._limit_suggestion_text(title, limit=300),
            "headings": self._limit_suggestion_text(" ".join(headings[:12]), limit=1200),
            "keywords": self._limit_suggestion_text(" ".join(keywords[:6]), limit=600),
            "abstract": abstract_text,
            "introduction": intro_text,
            "method": method_text,
            "contribution": contribution_text,
            "conclusion": conclusion_text,
            "lead": lead_text,
        }

    def _resolve_pdf_markdown_path(self, pdf_path: Path | None) -> Path | None:
        if pdf_path is None:
            return None
        pdf = Path(pdf_path).expanduser()
        candidate_roots: list[Path] = []
        seen: set[str] = set()

        def _push(raw: Path | str | None) -> None:
            if raw is None:
                return
            text = str(raw).strip()
            if not text:
                return
            try:
                root = Path(text).expanduser().resolve(strict=False)
            except Exception:
                root = Path(text).expanduser()
            key = str(root).lower()
            if key in seen:
                return
            seen.add(key)
            candidate_roots.append(root)

        _push(os.environ.get("KB_MD_DIR") or "")
        for prefs_path in (
            self._db_path.parent / "user_prefs.json",
            Path(__file__).resolve().parent.parent / "user_prefs.json",
        ):
            try:
                prefs = _load_local_prefs(prefs_path)
            except Exception:
                prefs = {}
            _push(str((prefs or {}).get("md_dir") or "").strip())
        _push(pdf.parent.parent / "md_output")
        _push(self._db_path.parent / "md_output")

        for root in candidate_roots:
            try:
                _md_folder, md_main, md_exists = _resolve_md_output_paths(root, pdf)
            except Exception:
                continue
            if md_exists:
                return md_main
        return None

    def _citation_meta_signal_parts(self, citation_meta: dict) -> dict[str, str]:
        parts: dict[str, str] = {
            "title": self._limit_suggestion_text(str(citation_meta.get("title") or ""), limit=400),
            "display_title": self._limit_suggestion_text(str(citation_meta.get("display_title") or ""), limit=400),
            "venue": self._limit_suggestion_text(str(citation_meta.get("venue") or ""), limit=160),
            "journal": self._limit_suggestion_text(str(citation_meta.get("journal") or ""), limit=160),
            "conference": self._limit_suggestion_text(str(citation_meta.get("conference") or ""), limit=160),
            "abstract": self._limit_suggestion_text(str(citation_meta.get("abstract") or ""), limit=2600),
            "summary": self._limit_suggestion_text(
                str(citation_meta.get("summary_line") or citation_meta.get("summary") or ""),
                limit=900,
            ),
        }
        for key in (
            "keywords",
            "keyword",
            "subjects",
            "subject",
            "topics",
            "topic",
            "fields_of_study",
            "field_of_study",
            "concepts",
            "concept",
        ):
            values = self._iter_nested_text(citation_meta.get(key), max_items=10)
            if values:
                parts[key] = self._limit_suggestion_text(" ".join(values), limit=900)
        authors = citation_meta.get("authors")
        if isinstance(authors, list):
            parts["authors"] = self._limit_suggestion_text(" ".join(str(item or "") for item in authors[:10]), limit=500)
        elif authors:
            parts["authors"] = self._limit_suggestion_text(str(authors), limit=500)
        return parts

    def _paper_suggestion_signals(self, row: sqlite3.Row | dict) -> list[dict[str, object]]:
        path_text = str(self._row_value(row, "path") or "")
        note_text = str(self._row_value(row, "note") or "")
        citation_meta = self._decode_citation_meta(str(self._row_value(row, "citation_meta") or ""))
        meta_parts = self._citation_meta_signal_parts(citation_meta)
        md_parts = self._extract_markdown_signal_parts(self._resolve_pdf_markdown_path(Path(path_text) if path_text else None))
        raw_signals = [
            ("path_stem", Path(path_text).stem if path_text else "", 1.4),
            ("path", path_text, 0.8),
            ("title", meta_parts.get("title") or "", 4.8),
            ("display_title", meta_parts.get("display_title") or "", 4.4),
            ("md_title", md_parts.get("title") or "", 5.2),
            ("summary", meta_parts.get("summary") or "", 3.4),
            ("abstract", meta_parts.get("abstract") or "", 3.8),
            ("md_abstract", md_parts.get("abstract") or "", 4.2),
            ("md_keywords", md_parts.get("keywords") or "", 4.0),
            ("keywords", meta_parts.get("keywords") or "", 3.4),
            ("subjects", meta_parts.get("subjects") or "", 3.2),
            ("topics", meta_parts.get("topics") or "", 3.2),
            ("fields_of_study", meta_parts.get("fields_of_study") or "", 3.0),
            ("concepts", meta_parts.get("concepts") or "", 3.0),
            ("headings", md_parts.get("headings") or "", 2.6),
            ("introduction", md_parts.get("introduction") or "", 2.6),
            ("method", md_parts.get("method") or "", 2.8),
            ("contribution", md_parts.get("contribution") or "", 3.0),
            ("conclusion", md_parts.get("conclusion") or "", 1.9),
            ("lead", md_parts.get("lead") or "", 1.8),
            ("venue", meta_parts.get("venue") or "", 1.0),
            ("journal", meta_parts.get("journal") or "", 1.0),
            ("conference", meta_parts.get("conference") or "", 1.0),
            ("authors", meta_parts.get("authors") or "", 0.6),
            ("note", note_text, 2.4),
        ]
        signals: list[dict[str, object]] = []
        for kind, text, weight in raw_signals:
            text_norm = self._normalize_suggestion_text(text)
            if not text_norm:
                continue
            signals.append(
                {
                    "kind": str(kind or ""),
                    "text_norm": text_norm,
                    "tokens": set(text_norm.split()),
                    "weight": float(weight),
                }
            )
        return signals

    def _has_generic_doc_type_evidence(self, signals: list[dict[str, object]], phrases: Iterable[str]) -> bool:
        for raw_phrase in phrases:
            phrase_norm = self._normalize_suggestion_text(raw_phrase)
            if not phrase_norm:
                continue
            for signal in signals:
                kind = str(signal.get("kind") or "")
                text_norm = str(signal.get("text_norm") or "")
                if kind not in _GENERIC_DOC_TYPE_SIGNAL_KINDS or not text_norm:
                    continue
                if self._text_contains_phrase(text_norm, phrase_norm):
                    return True
        return False

    def _generic_doc_type_allowed(self, signals: list[dict[str, object]], label: str) -> bool:
        label_clean = self._clean_category(label)
        if label_clean in _GENERIC_DOC_TYPE_CATEGORY_RULES:
            return self._has_generic_doc_type_evidence(signals, _GENERIC_DOC_TYPE_CATEGORY_RULES[label_clean])
        norm, _clean = self._normalize_tag(label)
        if norm in _GENERIC_DOC_TYPE_TAGS:
            category_label = norm.capitalize() if norm != "survey" else "Survey"
            return self._has_generic_doc_type_evidence(
                signals,
                _GENERIC_DOC_TYPE_CATEGORY_RULES.get(category_label, (norm,)),
            )
        return True

    def _category_rule_score(self, signals: list[dict[str, object]], label: str) -> float:
        label_clean = self._clean_category(label)
        score = self._score_label_match(signals, label_clean)
        phrases = _CATEGORY_SUGGESTION_RULES.get(label_clean) or _GENERIC_DOC_TYPE_CATEGORY_RULES.get(label_clean)
        if phrases:
            score = max(score, self._score_phrase_matches(signals, phrases))
        return score

    def _tag_rule_score(self, signals: list[dict[str, object]], label: str) -> float:
        _norm, clean = self._normalize_tag(label)
        score = self._score_label_match(signals, clean)
        phrases = _TAG_SUGGESTION_RULES.get(clean)
        if phrases:
            score = max(score, self._score_phrase_matches(signals, phrases))
        return score

    def _category_hint(self, label: str) -> str:
        return str(_SYSTEM_CATEGORY_HINTS.get(self._clean_category(label)) or "").strip()

    def _tag_hint(self, label: str) -> str:
        return str(_SYSTEM_TAG_HINTS.get(self._normalize_tag(label)[1]) or "").strip()

    def _canonicalize_category_candidate(self, value: str, candidate_labels: Iterable[str]) -> str:
        probe = self._normalize_suggestion_text(value)
        if not probe:
            return ""
        for label in candidate_labels:
            clean = self._clean_category(label)
            if clean and self._normalize_suggestion_text(clean) == probe:
                return clean
        return ""

    def _canonicalize_tag_candidate(self, value: str, candidate_labels: Iterable[str]) -> str:
        probe = self._normalize_suggestion_text(value)
        if not probe:
            return ""
        for label in candidate_labels:
            _norm, clean = self._normalize_tag(label)
            if clean and self._normalize_suggestion_text(clean) == probe:
                return clean
        return ""

    def _looks_like_low_quality_new_category(self, label: str) -> bool:
        clean = self._clean_category(label)
        norm = self._normalize_suggestion_text(clean)
        tokens = norm.split()
        if (not clean) or (not norm):
            return True
        if norm in _BAD_NEW_CATEGORY_NORMS:
            return True
        if len(tokens) > 4:
            return True
        if re.search(r"[,:;.!?]", clean):
            return True
        return False

    def _looks_like_low_quality_new_tag(self, label: str) -> bool:
        _norm, clean = self._normalize_tag(label)
        norm = self._normalize_suggestion_text(clean)
        tokens = norm.split()
        if (not clean) or (not norm):
            return True
        if norm in _BAD_NEW_TAG_NORMS:
            return True
        if len(tokens) > 4:
            return True
        if re.search(r"[,:;.!?]", clean):
            return True
        return False

    def _paper_text_for_suggestions(self, row: sqlite3.Row | dict) -> str:
        return " ".join(str(signal.get("text_norm") or "") for signal in self._paper_suggestion_signals(row))

    def _score_phrase_matches(self, signals: list[dict[str, object]], phrases: Iterable[str]) -> float:
        score = 0.0
        for raw_phrase in phrases:
            phrase_norm = self._normalize_suggestion_text(raw_phrase)
            if not phrase_norm:
                continue
            token_count = len(phrase_norm.split())
            phrase_weight = 0.9 + (max(0, token_count - 1) * 0.4)
            for signal in signals:
                text_norm = str(signal.get("text_norm") or "")
                if not text_norm or not self._text_contains_phrase(text_norm, phrase_norm):
                    continue
                score += float(signal.get("weight") or 0.0) * phrase_weight
        return score

    def _score_label_match(self, signals: list[dict[str, object]], label: str) -> float:
        label_norm = self._normalize_suggestion_text(label)
        if not label_norm:
            return 0.0
        label_tokens = set(label_norm.split())
        if not label_tokens:
            return 0.0

        score = 0.0
        exact_multiplier = 1.2 + (max(0, len(label_tokens) - 1) * 0.35)
        best_partial = 0.0
        for signal in signals:
            text_norm = str(signal.get("text_norm") or "")
            tokens = signal.get("tokens")
            weight = float(signal.get("weight") or 0.0)
            if not text_norm or not isinstance(tokens, set):
                continue
            if self._text_contains_phrase(text_norm, label_norm):
                score += weight * exact_multiplier
                continue
            overlap = len(tokens & label_tokens)
            if overlap <= 0:
                continue
            coverage = overlap / max(1, len(label_tokens))
            if len(label_tokens) == 1:
                continue
            if overlap >= min(2, len(label_tokens)) and coverage >= 0.6:
                best_partial = max(best_partial, weight * (0.45 + coverage))
        return score + best_partial

    def _list_user_taxonomy_stats(self, conn: sqlite3.Connection) -> dict[str, object]:
        category_rows = conn.execute(
            """
            SELECT paper_category, COUNT(*) AS freq
            FROM paper_meta
            WHERE paper_category <> ''
            GROUP BY paper_category
            ORDER BY freq DESC, LOWER(paper_category), paper_category
            """
        ).fetchall()
        categories: dict[str, int] = {}
        for row in category_rows:
            label = self._clean_category(str(row["paper_category"] or ""))
            if not label:
                continue
            try:
                categories[label] = int(row["freq"] or 0)
            except Exception:
                categories[label] = 0

        tag_rows = conn.execute(
            """
            SELECT tag_norm, MIN(tag_label) AS tag_label, COUNT(*) AS freq
            FROM paper_tags
            WHERE source = 'user'
            GROUP BY tag_norm
            ORDER BY freq DESC, LOWER(MIN(tag_label)), MIN(tag_label)
            """
        ).fetchall()
        tags: dict[str, dict[str, object]] = {}
        for row in tag_rows:
            norm, label = self._normalize_tag(str(row["tag_label"] or ""))
            if not norm or not label:
                continue
            try:
                freq = int(row["freq"] or 0)
            except Exception:
                freq = 0
            tags[norm] = {"label": label, "count": freq}

        category_tags: dict[str, dict[str, dict[str, object]]] = {}
        tag_categories: dict[str, dict[str, int]] = {}
        category_tag_rows = conn.execute(
            """
            SELECT pm.paper_category, pt.tag_norm, MIN(pt.tag_label) AS tag_label, COUNT(*) AS freq
            FROM paper_meta pm
            JOIN paper_tags pt ON pt.sha1 = pm.sha1
            WHERE pm.paper_category <> '' AND pt.source = 'user'
            GROUP BY pm.paper_category, pt.tag_norm
            ORDER BY freq DESC, LOWER(pm.paper_category), LOWER(MIN(pt.tag_label))
            """
        ).fetchall()
        for row in category_tag_rows:
            category = self._clean_category(str(row["paper_category"] or ""))
            norm, label = self._normalize_tag(str(row["tag_label"] or ""))
            if not category or not norm or not label:
                continue
            try:
                freq = int(row["freq"] or 0)
            except Exception:
                freq = 0
            category_tags.setdefault(category, {})[norm] = {"label": label, "count": freq}
            tag_categories.setdefault(norm, {})[category] = freq

        return {
            "categories": categories,
            "tags": tags,
            "category_tags": category_tags,
            "tag_categories": tag_categories,
        }

    def _env_flag(self, name: str, *, default: bool) -> bool:
        raw = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
        return raw not in {"0", "false", "off", "no"}

    def _env_int(self, name: str, *, default: int, lo: int, hi: int) -> int:
        try:
            value = int(str(os.environ.get(name, default) or default).strip())
        except Exception:
            value = int(default)
        return max(int(lo), min(int(hi), value))

    def _coerce_float(self, value: object, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _extract_json_block(self, text: str) -> dict:
        raw = str(text or "").strip()
        if not raw:
            return {}
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()
        match = _LLM_JSON_BLOCK_RE.search(raw)
        if match:
            raw = match.group(0).strip()
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _prompt_text_excerpt(self, value: str | None, *, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return ""
        return text[: max(1, int(limit))]

    def _build_suggestion_llm(self, *, total_targets: int):
        if not self._env_flag("KB_LIBRARY_SUGGEST_USE_LLM", default=True):
            return None
        max_targets = self._env_int("KB_LIBRARY_SUGGEST_LLM_MAX_TARGETS", default=12, lo=1, hi=200)
        if int(total_targets or 0) > max_targets:
            return None
        try:
            from kb.config import load_settings
            from kb.llm import DeepSeekChat
        except Exception:
            return None
        try:
            settings = load_settings()
        except Exception:
            return None
        if not getattr(settings, "api_key", None):
            return None
        try:
            fast_settings = replace(
                settings,
                timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 12.0),
                max_retries=0,
            )
        except Exception:
            fast_settings = settings
        try:
            return DeepSeekChat(fast_settings)
        except Exception:
            return None

    def _top_category_candidates_for_llm(
        self,
        *,
        signals: list[dict[str, object]],
        taxonomy_stats: dict[str, object] | None,
        current_category: str,
        dismissed_category: str,
        heuristic_category: str,
        limit: int = 18,
    ) -> list[dict[str, object]]:
        category_stats = (taxonomy_stats or {}).get("categories")
        if not isinstance(category_stats, dict):
            category_stats = {}
        pooled: dict[str, dict[str, object]] = {}
        for label, freq in category_stats.items():
            clean = self._clean_category(str(label or ""))
            if not clean or clean == current_category or clean == dismissed_category:
                continue
            pooled[clean] = {
                "label": clean,
                "count": int(freq or 0),
                "source": "user",
            }
        for label in list(_CATEGORY_SUGGESTION_RULES.keys()) + list(_GENERIC_DOC_TYPE_CATEGORY_RULES.keys()):
            clean = self._clean_category(label)
            if (not clean) or clean == current_category or clean == dismissed_category:
                continue
            pooled.setdefault(
                clean,
                {
                    "label": clean,
                    "count": 0,
                    "source": "system",
                },
            )

        ranked: list[tuple[float, int, str, str]] = []
        for clean, meta in pooled.items():
            if (clean in _GENERIC_DOC_TYPE_CATEGORY_LABELS) and (not self._generic_doc_type_allowed(signals, clean)):
                continue
            score = self._category_rule_score(signals, clean)
            score += min(1.6, max(0, int(meta.get("count") or 0) - 1) * 0.2)
            if clean == heuristic_category:
                score += 2.4
            if score <= 0.0 and clean != heuristic_category:
                continue
            ranked.append((score, int(meta.get("count") or 0), clean, str(meta.get("source") or "user")))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].lower()))
        out: list[dict[str, object]] = []
        for score, freq, label, source in ranked[: max(1, int(limit))]:
            out.append(
                {
                    "label": label,
                    "count": int(freq),
                    "score": round(float(score), 3),
                    "source": source,
                    "hint": self._category_hint(label),
                }
            )
        return out

    def _top_tag_candidates_for_llm(
        self,
        *,
        signals: list[dict[str, object]],
        taxonomy_stats: dict[str, object] | None,
        current_user_tags: Iterable[str] | None,
        dismissed_tags: Iterable[str] | None,
        heuristic_tags: Iterable[str] | None,
        limit: int = 40,
    ) -> list[dict[str, object]]:
        tag_stats = (taxonomy_stats or {}).get("tags")
        if not isinstance(tag_stats, dict):
            tag_stats = {}
        current_norms = {
            self._normalize_tag(tag)[0]
            for tag in (current_user_tags or [])
            if self._normalize_tag(tag)[0]
        }
        dismissed_norms = {
            self._normalize_tag(tag)[0]
            for tag in (dismissed_tags or [])
            if self._normalize_tag(tag)[0]
        }
        heuristic_norms = {
            self._normalize_tag(tag)[0]
            for tag in (heuristic_tags or [])
            if self._normalize_tag(tag)[0]
        }
        pooled: dict[str, dict[str, object]] = {}
        for norm, info in tag_stats.items():
            if (not norm) or (norm in current_norms) or (norm in dismissed_norms) or (not isinstance(info, dict)):
                continue
            label = self._normalize_tag(str(info.get("label") or ""))[1]
            if not label:
                continue
            pooled[norm] = {
                "label": label,
                "count": int(info.get("count") or 0),
                "source": "user",
            }
        for label in _TAG_SUGGESTION_RULES.keys():
            norm, clean = self._normalize_tag(label)
            if (not norm) or (norm in current_norms) or (norm in dismissed_norms):
                continue
            pooled.setdefault(
                norm,
                {
                    "label": clean,
                    "count": 0,
                    "source": "system",
                },
            )

        ranked: list[tuple[float, int, str, str, str]] = []
        for norm, meta in pooled.items():
            label = self._normalize_tag(str(meta.get("label") or ""))[1]
            if not label:
                continue
            if (norm in _GENERIC_DOC_TYPE_TAGS) and (not self._generic_doc_type_allowed(signals, label)):
                continue
            score = self._tag_rule_score(signals, label)
            score += min(1.2, max(0, int(meta.get("count") or 0) - 1) * 0.15)
            if norm in heuristic_norms:
                score += 1.8
            if score <= 0.0 and norm not in heuristic_norms:
                continue
            ranked.append((score, int(meta.get("count") or 0), norm, label, str(meta.get("source") or "user")))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[3].lower()))
        out: list[dict[str, object]] = []
        for score, freq, _norm, label, source in ranked[: max(1, int(limit))]:
            out.append(
                {
                    "label": label,
                    "count": int(freq),
                    "score": round(float(score), 3),
                    "source": source,
                    "hint": self._tag_hint(label),
                }
            )
        return out

    def _build_llm_paper_payload(
        self,
        *,
        row: sqlite3.Row | dict,
        heuristic_category: str,
        heuristic_tags: Iterable[str] | None,
    ) -> dict[str, object]:
        path_text = str(self._row_value(row, "path") or "")
        note_text = str(self._row_value(row, "note") or "")
        citation_meta = self._decode_citation_meta(str(self._row_value(row, "citation_meta") or ""))
        meta_parts = self._citation_meta_signal_parts(citation_meta)
        md_parts = self._extract_markdown_signal_parts(self._resolve_pdf_markdown_path(Path(path_text) if path_text else None))
        title = (
            self._prompt_text_excerpt(md_parts.get("title") or "", limit=320)
            or self._prompt_text_excerpt(meta_parts.get("title") or meta_parts.get("display_title") or "", limit=320)
            or Path(path_text).stem
        )
        return {
            "path_stem": Path(path_text).stem if path_text else "",
            "title": title,
            "summary_line": self._prompt_text_excerpt(meta_parts.get("summary") or "", limit=420),
            "abstract": self._prompt_text_excerpt(md_parts.get("abstract") or meta_parts.get("abstract") or "", limit=1400),
            "keywords": self._prompt_text_excerpt(md_parts.get("keywords") or meta_parts.get("keywords") or "", limit=360),
            "headings": self._prompt_text_excerpt(md_parts.get("headings") or "", limit=700),
            "introduction": self._prompt_text_excerpt(md_parts.get("introduction") or md_parts.get("lead") or "", limit=1000),
            "method": self._prompt_text_excerpt(md_parts.get("method") or md_parts.get("contribution") or "", limit=1000),
            "conclusion": self._prompt_text_excerpt(md_parts.get("conclusion") or "", limit=700),
            "note": self._prompt_text_excerpt(note_text, limit=600),
            "venue": self._prompt_text_excerpt(meta_parts.get("venue") or meta_parts.get("journal") or meta_parts.get("conference") or "", limit=160),
            "heuristic_category": self._clean_category(heuristic_category),
            "heuristic_tags": [self._normalize_tag(tag)[1] for tag in (heuristic_tags or []) if self._normalize_tag(tag)[1]],
        }

    def _generate_llm_suggestions_for_row(
        self,
        *,
        llm_client,
        row: sqlite3.Row | dict,
        current_user_tags: Iterable[str] | None,
        dismissed_tags: Iterable[str] | None,
        taxonomy_stats: dict[str, object] | None,
        dismissed_category: str,
        heuristic_category: str,
        heuristic_tags: Iterable[str] | None,
    ) -> tuple[str, list[str]]:
        if llm_client is None:
            return "", []
        signals = self._paper_suggestion_signals(row)
        if not signals:
            return "", []

        current_category = self._clean_category(str(self._row_value(row, "paper_category") or ""))
        dismissed_category_clean = self._clean_category(dismissed_category)
        category_candidates = self._top_category_candidates_for_llm(
            signals=signals,
            taxonomy_stats=taxonomy_stats,
            current_category=current_category,
            dismissed_category=dismissed_category_clean,
            heuristic_category=self._clean_category(heuristic_category),
        )
        tag_candidates = self._top_tag_candidates_for_llm(
            signals=signals,
            taxonomy_stats=taxonomy_stats,
            current_user_tags=current_user_tags,
            dismissed_tags=dismissed_tags,
            heuristic_tags=heuristic_tags,
        )

        category_tag_stats = (taxonomy_stats or {}).get("category_tags")
        if not isinstance(category_tag_stats, dict):
            category_tag_stats = {}
        preferred_category = current_category or self._clean_category(heuristic_category)
        preferred_tag_hints: list[str] = []
        if preferred_category:
            tag_map = category_tag_stats.get(preferred_category)
            if isinstance(tag_map, dict):
                ranked_hints = sorted(
                    (
                        (int(info.get("count") or 0), self._normalize_tag(str(info.get("label") or ""))[1])
                        for info in tag_map.values()
                        if isinstance(info, dict) and self._normalize_tag(str(info.get("label") or ""))[1]
                    ),
                    key=lambda item: (-item[0], item[1].lower()),
                )
                preferred_tag_hints = [label for _count, label in ranked_hints[:10]]

        paper_payload = self._build_llm_paper_payload(
            row=row,
            heuristic_category=heuristic_category,
            heuristic_tags=heuristic_tags,
        )
        prompt_payload = {
            "paper": paper_payload,
            "current_category": current_category,
            "current_user_tags": [self._normalize_tag(tag)[1] for tag in (current_user_tags or []) if self._normalize_tag(tag)[1]],
            "dismissed_category": dismissed_category_clean,
            "dismissed_tags": [self._normalize_tag(tag)[1] for tag in (dismissed_tags or []) if self._normalize_tag(tag)[1]],
            "existing_categories": category_candidates,
            "existing_tags": tag_candidates,
            "preferred_tags_for_category": preferred_tag_hints,
        }
        system = (
            "You assign research-paper taxonomy suggestions for a private library.\n"
            "Return JSON only with keys: suggested_category, suggested_tags, category_confidence, tag_confidence, reason.\n"
            "Rules:\n"
            "- Use only the provided paper evidence and library taxonomy.\n"
            "- suggested_category must be one stable topical home for the paper, not a vague area.\n"
            "- Prefer one of the provided category candidates. Use empty string if none fits well.\n"
            "- Tags are reusable normalized library facets, closer to curated keywords than raw abstract phrases.\n"
            "- Good tags describe modality, task, constraint, or method property.\n"
            "- Bad tags are generic words like method, model, framework, imaging, deep-learning, dataset, benchmark, survey.\n"
            "- suggested_tags must be an array of 0 to 6 short tags.\n"
            "- Prefer the provided tag candidates. Use empty array if none fit well.\n"
            "- Reuse the user's wording when a similar existing label already exists.\n"
            "- Avoid duplicate or near-duplicate tags, and avoid tags that merely repeat the category.\n"
            "- Do not output the dismissed category or dismissed/current tags.\n"
            "- Confidence values must be between 0 and 1.\n"
            "- reason must be one short sentence.\n"
        )
        try:
            raw = (
                llm_client.chat(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
                    ],
                    temperature=0.0,
                    max_tokens=420,
                )
                or ""
            ).strip()
        except Exception:
            return "", []

        data = self._extract_json_block(raw)
        if not data:
            return "", []

        category_conf = self._coerce_float(data.get("category_confidence"), default=0.0)
        tag_conf = self._coerce_float(data.get("tag_confidence"), default=0.0)
        allow_new_category = self._env_flag("KB_LIBRARY_SUGGEST_ALLOW_NEW_CATEGORY", default=False)
        allow_new_tags = self._env_flag("KB_LIBRARY_SUGGEST_ALLOW_NEW_TAGS", default=False)
        category_candidate_labels = [str(item.get("label") or "") for item in category_candidates if isinstance(item, dict)]
        tag_candidate_labels = [str(item.get("label") or "") for item in tag_candidates if isinstance(item, dict)]

        suggested_category_raw = self._clean_category(str(data.get("suggested_category") or ""))
        suggested_category = self._canonicalize_category_candidate(suggested_category_raw, category_candidate_labels) or suggested_category_raw
        if suggested_category in {current_category, dismissed_category_clean}:
            suggested_category = ""
        if suggested_category:
            if not self._generic_doc_type_allowed(signals, suggested_category):
                suggested_category = ""
            else:
                is_known_category = suggested_category in set(category_candidate_labels)
                if (not is_known_category) and (not allow_new_category):
                    suggested_category = ""
                elif (not is_known_category) and self._looks_like_low_quality_new_category(suggested_category):
                    suggested_category = ""
                else:
                    min_category_conf = 0.45 if is_known_category else (0.65 if not category_candidate_labels else 0.84)
                    if category_conf < min_category_conf:
                        suggested_category = ""

        known_tag_norms = {
            self._normalize_tag(str(item.get("label") or ""))[0]
            for item in tag_candidates
            if isinstance(item, dict) and self._normalize_tag(str(item.get("label") or ""))[0]
        }
        current_tag_norms = {
            self._normalize_tag(tag)[0]
            for tag in (current_user_tags or [])
            if self._normalize_tag(tag)[0]
        }
        dismissed_tag_norms = {
            self._normalize_tag(tag)[0]
            for tag in (dismissed_tags or [])
            if self._normalize_tag(tag)[0]
        }
        cleaned_tags: list[str] = []
        new_tag_budget = 1 if allow_new_tags else 0
        for raw_tag in self._clean_tags(data.get("suggested_tags") or []):
            canonical = self._canonicalize_tag_candidate(raw_tag, tag_candidate_labels) or raw_tag
            norm, label = self._normalize_tag(canonical)
            if (not norm) or (norm in current_tag_norms) or (norm in dismissed_tag_norms):
                continue
            if not self._generic_doc_type_allowed(signals, label):
                continue
            if norm in known_tag_norms:
                if tag_conf < 0.45:
                    continue
            else:
                if (not allow_new_tags) or self._looks_like_low_quality_new_tag(label):
                    continue
                if tag_conf < 0.8 or new_tag_budget <= 0:
                    continue
                new_tag_budget -= 1
            if norm not in {self._normalize_tag(item)[0] for item in cleaned_tags}:
                cleaned_tags.append(label)
            if len(cleaned_tags) >= 6:
                break
        return suggested_category, cleaned_tags

    def _merge_tag_suggestions(
        self,
        *,
        primary: Iterable[str] | None,
        secondary: Iterable[str] | None,
        current_user_tags: Iterable[str] | None,
        dismissed_tags: Iterable[str] | None,
        limit: int = 6,
    ) -> list[str]:
        current_norms = {
            self._normalize_tag(tag)[0]
            for tag in (current_user_tags or [])
            if self._normalize_tag(tag)[0]
        }
        dismissed_norms = {
            self._normalize_tag(tag)[0]
            for tag in (dismissed_tags or [])
            if self._normalize_tag(tag)[0]
        }
        out: list[str] = []
        seen: set[str] = set()
        for source in (primary or []), (secondary or []):
            for raw_tag in source:
                norm, label = self._normalize_tag(raw_tag)
                if (
                    (not norm)
                    or (norm in seen)
                    or (norm in current_norms)
                    or (norm in dismissed_norms)
                ):
                    continue
                seen.add(norm)
                out.append(label)
                if len(out) >= max(1, int(limit)):
                    return out
        return out

    def _generate_suggestions_for_row(
        self,
        *,
        row: sqlite3.Row | dict,
        current_user_tags: Iterable[str] | None,
        dismissed_tags: Iterable[str] | None,
        taxonomy_stats: dict[str, object] | None,
        dismissed_category: str,
    ) -> tuple[str, list[str]]:
        signals = self._paper_suggestion_signals(row)
        if not signals:
            return "", []

        user_tag_norms = {
            self._normalize_tag(tag)[0]
            for tag in (current_user_tags or [])
            if self._normalize_tag(tag)[0]
        }
        dismissed_tag_norms = {self._normalize_tag(tag)[0] for tag in (dismissed_tags or []) if self._normalize_tag(tag)[0]}
        dismissed_category_clean = self._clean_category(dismissed_category)
        current_category = self._clean_category(str(self._row_value(row, "paper_category") or ""))
        category_stats = (taxonomy_stats or {}).get("categories")
        if not isinstance(category_stats, dict):
            category_stats = {}
        tag_stats = (taxonomy_stats or {}).get("tags")
        if not isinstance(tag_stats, dict):
            tag_stats = {}
        category_tag_stats = (taxonomy_stats or {}).get("category_tags")
        if not isinstance(category_tag_stats, dict):
            category_tag_stats = {}
        tag_category_stats = (taxonomy_stats or {}).get("tag_categories")
        if not isinstance(tag_category_stats, dict):
            tag_category_stats = {}

        category_scores: dict[str, float] = {}
        for label, phrases in _CATEGORY_SUGGESTION_RULES.items():
            score = self._score_phrase_matches(signals, phrases)
            if score > 0:
                category_scores[label] = max(category_scores.get(label, 0.0), score)
        for label, phrases in _GENERIC_DOC_TYPE_CATEGORY_RULES.items():
            if not self._has_generic_doc_type_evidence(signals, phrases):
                continue
            score = self._score_phrase_matches(signals, phrases)
            if score > 0:
                category_scores[label] = max(category_scores.get(label, 0.0), score)
        for label, freq in category_stats.items():
            label_clean = self._clean_category(str(label or ""))
            if not label_clean or label_clean == current_category:
                continue
            if (label_clean in _GENERIC_DOC_TYPE_CATEGORY_LABELS) and (not self._generic_doc_type_allowed(signals, label_clean)):
                continue
            score = self._score_label_match(signals, label_clean)
            if score > 0:
                category_scores[label_clean] = max(
                    category_scores.get(label_clean, 0.0),
                    score + min(1.8, max(0, int(freq or 0) - 1) * 0.25),
                )
        for tag_norm in user_tag_norms:
            category_hits = tag_category_stats.get(tag_norm)
            if not isinstance(category_hits, dict):
                continue
            for category_label, freq in category_hits.items():
                label_clean = self._clean_category(str(category_label or ""))
                if not label_clean or label_clean == current_category:
                    continue
                category_scores[label_clean] = category_scores.get(label_clean, 0.0) + min(
                    2.4,
                    0.75 * max(1, int(freq or 0)),
                )

        suggested_category = ""
        if not current_category and category_scores:
            ranked_categories = sorted(category_scores.items(), key=lambda item: (-float(item[1]), item[0].lower()))
            for label, score in ranked_categories:
                if label != dismissed_category_clean:
                    if float(score) < 3.2:
                        break
                    suggested_category = label
                    break

        tag_scores: dict[str, float] = {}
        for label, phrases in _TAG_SUGGESTION_RULES.items():
            score = self._score_phrase_matches(signals, phrases)
            if score > 0:
                tag_scores[label] = max(tag_scores.get(label, 0.0), score)
        for norm, info in tag_stats.items():
            if not isinstance(info, dict):
                continue
            clean = self._normalize_tag(str(info.get("label") or ""))[1]
            if not clean or not norm:
                continue
            if (norm in _GENERIC_DOC_TYPE_TAGS) and (not self._generic_doc_type_allowed(signals, clean)):
                continue
            score = self._score_label_match(signals, clean)
            if score > 0:
                tag_scores[clean] = max(
                    tag_scores.get(clean, 0.0),
                    score + min(1.5, max(0, int(info.get("count") or 0) - 1) * 0.2),
                )

        effective_category = current_category or suggested_category
        if effective_category:
            category_tag_hits = category_tag_stats.get(effective_category)
            if isinstance(category_tag_hits, dict):
                for norm, info in category_tag_hits.items():
                    if not isinstance(info, dict):
                        continue
                    clean = self._normalize_tag(str(info.get("label") or ""))[1]
                    if not clean or not norm:
                        continue
                    if (norm in _GENERIC_DOC_TYPE_TAGS) and (not self._generic_doc_type_allowed(signals, clean)):
                        continue
                    base_score = self._score_label_match(signals, clean)
                    if base_score <= 0:
                        continue
                    tag_scores[clean] = max(
                        tag_scores.get(clean, 0.0),
                        base_score + min(2.0, max(1, int(info.get("count") or 0)) * 0.4),
                    )

        ranked_tags = sorted(tag_scores.items(), key=lambda item: (-float(item[1]), item[0].lower()))
        suggested_tags: list[str] = []
        seen_tag_norms: set[str] = set()
        for label, score in ranked_tags:
            norm, clean = self._normalize_tag(label)
            if (
                not norm
                or norm in seen_tag_norms
                or norm in user_tag_norms
                or norm in dismissed_tag_norms
            ):
                continue
            if float(score) < 1.8:
                continue
            seen_tag_norms.add(norm)
            suggested_tags.append(clean)
            if len(suggested_tags) >= 6:
                break

        return suggested_category, suggested_tags

    def _fetch_tags_by_source(self, conn: sqlite3.Connection, sha1s: list[str], source: str) -> dict[str, list[str]]:
        if not sha1s:
            return {}
        qmarks = ",".join("?" for _ in sha1s)
        rows = conn.execute(
            f"""
            SELECT sha1, tag_label
            FROM paper_tags
            WHERE source = ? AND sha1 IN ({qmarks})
            ORDER BY LOWER(tag_label), tag_label
            """,
            (source, *sha1s),
        ).fetchall()
        out: dict[str, list[str]] = {}
        for row in rows:
            sha1 = str(row["sha1"] or "")
            label = str(row["tag_label"] or "").strip()
            if not sha1 or not label:
                continue
            out.setdefault(sha1, []).append(label)
        return out

    def _clean_category(self, value: str | None) -> str:
        return str(value or "").strip().replace("\r", " ").replace("\n", " ")[:80]

    def _clean_note(self, value: str | None) -> str:
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = text.strip()
        return text[:4000]

    def _clean_reading_status(self, value: str | None) -> str:
        status = str(value or "").strip().lower()
        return status if status in _READING_STATUS_SET else ""

    def _normalize_tag(self, value: str | None) -> tuple[str, str]:
        label = " ".join(str(value or "").strip().replace(",", " ").split())
        if not label:
            return "", ""
        label = label[:40]
        norm = label.lower()
        return norm, label

    def _clean_tags(self, values: Iterable[str] | None) -> list[str]:
        if not values:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            norm, label = self._normalize_tag(raw)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(label)
            if len(out) >= 16:
                break
        return out

    def get_by_sha1(self, sha1: str) -> dict | None:
        sha1 = (sha1 or "").strip().lower()
        if not sha1:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT sha1, path, created_at, citation_meta FROM pdf_files WHERE sha1 = ?", (sha1,)).fetchone()
        return dict(row) if row else None
    
    def get_by_path(self, path: Path) -> dict | None:
        """Get PDF record by path (for citation lookup)."""
        path_s = str(Path(path))
        with self._connect() as conn:
            row = conn.execute("SELECT sha1, path, created_at, citation_meta FROM pdf_files WHERE path = ?", (path_s,)).fetchone()
        return dict(row) if row else None
    
    def get_citation_meta(self, path: Path) -> dict | None:
        """Get stored citation metadata for a PDF path."""
        record = self.get_by_path(path)
        if not record or not record.get("citation_meta"):
            return None
        try:
            return json.loads(record["citation_meta"])
        except Exception:
            return None
    
    def set_citation_meta(self, path: Path, citation_meta: dict | None) -> None:
        """Store citation metadata for a PDF path."""
        path_s = str(Path(path))
        meta_json = json.dumps(citation_meta) if citation_meta else None
        with self._connect() as conn:
            conn.execute(
                "UPDATE pdf_files SET citation_meta = ? WHERE path = ?",
                (meta_json, path_s)
            )

    def upsert(self, sha1: str, path: Path, citation_meta: dict | None = None) -> None:
        sha1 = (sha1 or "").strip().lower()
        path_s = str(Path(path))
        now = time.time()
        meta_json = json.dumps(citation_meta) if citation_meta else None
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO pdf_files (sha1, path, created_at, citation_meta) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(sha1) DO UPDATE SET path=excluded.path, citation_meta=excluded.citation_meta",
                (sha1, path_s, now, meta_json),
            )

    def ensure_record_for_path(self, path: Path | str) -> dict | None:
        path_obj = Path(path)
        rec = self.get_by_path(path_obj)
        if rec:
            return rec
        try:
            if not path_obj.exists() or not path_obj.is_file():
                return None
            sha1 = _sha1_bytes(path_obj.read_bytes())
            self.upsert(sha1, path_obj)
            return self.get_by_path(path_obj)
        except Exception:
            return None

    def list_records_by_paths(self, paths: list[Path]) -> dict[str, dict]:
        wanted = [str(Path(path)) for path in paths if str(path or "").strip()]
        if not wanted:
            return {}
        qmarks = ",".join("?" for _ in wanted)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                  pf.sha1,
                  pf.path,
                  pf.created_at,
                  pf.citation_meta,
                  COALESCE(pm.paper_category, '') AS paper_category,
                  COALESCE(pm.reading_status, '') AS reading_status,
                  COALESCE(pm.note, '') AS note,
                  COALESCE(pm.suggested_category, '') AS suggested_category,
                  COALESCE(pm.dismissed_category, '') AS dismissed_category
                FROM pdf_files pf
                LEFT JOIN paper_meta pm ON pm.sha1 = pf.sha1
                WHERE pf.path IN ({qmarks})
                """,
                wanted,
            ).fetchall()
            sha1s = [str(row["sha1"] or "") for row in rows if str(row["sha1"] or "").strip()]
            user_tags_by_sha1 = self._fetch_tags_by_source(conn, sha1s, "user")
            suggested_tags_by_sha1 = self._fetch_tags_by_source(conn, sha1s, "suggested")
        out: dict[str, dict] = {}
        for row in rows:
            rec = dict(row)
            sha1 = str(rec.get("sha1") or "")
            rec["user_tags"] = list(user_tags_by_sha1.get(sha1, []))
            rec["suggested_tags"] = list(suggested_tags_by_sha1.get(sha1, []))
            rec["has_suggestions"] = bool(rec.get("suggested_category") or rec.get("suggested_tags"))
            out[str(rec.get("path") or "")] = rec
        return out

    def get_paper_user_meta(self, *, sha1: str = "", path: Path | str | None = None) -> dict | None:
        sha1_clean = str(sha1 or "").strip().lower()
        if not sha1_clean and path is not None:
            rec = self.ensure_record_for_path(path)
            sha1_clean = str((rec or {}).get("sha1") or "").strip().lower()
        if not sha1_clean:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  pf.sha1,
                  pf.path,
                  pf.citation_meta,
                  COALESCE(pm.paper_category, '') AS paper_category,
                  COALESCE(pm.reading_status, '') AS reading_status,
                  COALESCE(pm.note, '') AS note,
                  COALESCE(pm.suggested_category, '') AS suggested_category,
                  COALESCE(pm.dismissed_category, '') AS dismissed_category
                FROM pdf_files pf
                LEFT JOIN paper_meta pm ON pm.sha1 = pf.sha1
                WHERE pf.sha1 = ?
                """,
                (sha1_clean,),
            ).fetchone()
            if not row:
                return None
            user_tag_rows = conn.execute(
                """
                SELECT tag_label
                FROM paper_tags
                WHERE sha1 = ? AND source = 'user'
                ORDER BY LOWER(tag_label), tag_label
                """,
                (sha1_clean,),
            ).fetchall()
            suggested_tag_rows = conn.execute(
                """
                SELECT tag_label
                FROM paper_tags
                WHERE sha1 = ? AND source = 'suggested'
                ORDER BY LOWER(tag_label), tag_label
                """,
                (sha1_clean,),
            ).fetchall()
            dismissed_tag_rows = conn.execute(
                """
                SELECT tag_label
                FROM paper_tags
                WHERE sha1 = ? AND source = 'dismissed'
                ORDER BY LOWER(tag_label), tag_label
                """,
                (sha1_clean,),
            ).fetchall()
        payload = dict(row)
        payload["user_tags"] = [str(tag_row["tag_label"] or "").strip() for tag_row in user_tag_rows if str(tag_row["tag_label"] or "").strip()]
        payload["suggested_tags"] = [str(tag_row["tag_label"] or "").strip() for tag_row in suggested_tag_rows if str(tag_row["tag_label"] or "").strip()]
        payload["dismissed_tags"] = [str(tag_row["tag_label"] or "").strip() for tag_row in dismissed_tag_rows if str(tag_row["tag_label"] or "").strip()]
        payload["has_suggestions"] = bool(payload.get("suggested_category") or payload.get("suggested_tags"))
        return payload

    def upsert_paper_user_meta(
        self,
        *,
        sha1: str = "",
        path: Path | str | None = None,
        paper_category: str = "",
        reading_status: str = "",
        note: str = "",
        user_tags: Iterable[str] | None = None,
    ) -> dict | None:
        sha1_clean = str(sha1 or "").strip().lower()
        if not sha1_clean and path is not None:
            rec = self.ensure_record_for_path(path)
            sha1_clean = str((rec or {}).get("sha1") or "").strip().lower()
        if not sha1_clean:
            return None

        category_clean = self._clean_category(paper_category)
        status_clean = self._clean_reading_status(reading_status)
        note_clean = self._clean_note(note)
        tags_clean = self._clean_tags(user_tags)
        now = time.time()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_meta (sha1, paper_category, reading_status, note, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(sha1) DO UPDATE SET
                  paper_category = excluded.paper_category,
                  reading_status = excluded.reading_status,
                  note = excluded.note,
                  updated_at = excluded.updated_at
                """,
                (sha1_clean, category_clean, status_clean, note_clean, now),
            )
            conn.execute(
                "DELETE FROM paper_tags WHERE sha1 = ? AND source = 'user'",
                (sha1_clean,),
            )
            for tag in tags_clean:
                norm, label = self._normalize_tag(tag)
                if not norm or not label:
                    continue
                conn.execute(
                    """
                    INSERT INTO paper_tags (sha1, tag_norm, tag_label, source, updated_at)
                    VALUES (?, ?, ?, 'user', ?)
                    """,
                    (sha1_clean, norm, label, now),
                )
            conn.execute(
                """
                UPDATE paper_meta
                SET
                  suggested_category = CASE WHEN suggested_category = ? THEN '' ELSE suggested_category END,
                  dismissed_category = CASE WHEN dismissed_category = ? THEN '' ELSE dismissed_category END
                WHERE sha1 = ?
                """,
                (category_clean, category_clean, sha1_clean),
            )
            if tags_clean:
                tag_norms = [self._normalize_tag(tag)[0] for tag in tags_clean if self._normalize_tag(tag)[0]]
                if tag_norms:
                    qmarks = ",".join("?" for _ in tag_norms)
                    conn.execute(
                        f"""
                        DELETE FROM paper_tags
                        WHERE sha1 = ? AND source IN ('suggested', 'dismissed') AND tag_norm IN ({qmarks})
                        """,
                        (sha1_clean, *tag_norms),
                    )
        return self.get_paper_user_meta(sha1=sha1_clean)

    def regenerate_paper_suggestions(
        self,
        *,
        sha1s: Iterable[str] | None = None,
        paths: Iterable[Path | str] | None = None,
    ) -> list[dict]:
        targets: list[str] = []
        seen: set[str] = set()

        for raw_sha1 in sha1s or []:
            sha1_clean = str(raw_sha1 or "").strip().lower()
            if not sha1_clean or sha1_clean in seen:
                continue
            seen.add(sha1_clean)
            targets.append(sha1_clean)

        for raw_path in paths or []:
            rec = self.ensure_record_for_path(raw_path)
            sha1_clean = str((rec or {}).get("sha1") or "").strip().lower()
            if not sha1_clean or sha1_clean in seen:
                continue
            seen.add(sha1_clean)
            targets.append(sha1_clean)

        with self._connect() as conn:
            if not targets:
                rows = conn.execute("SELECT sha1 FROM pdf_files ORDER BY created_at DESC").fetchall()
                targets = [str(row["sha1"] or "").strip().lower() for row in rows if str(row["sha1"] or "").strip()]

            taxonomy_stats = self._list_user_taxonomy_stats(conn)
            llm_client = self._build_suggestion_llm(total_targets=len(targets))
            now = time.time()
            updated_items: list[dict] = []

            for sha1_clean in targets:
                row = conn.execute(
                    """
                    SELECT
                      pf.sha1,
                      pf.path,
                      pf.citation_meta,
                      COALESCE(pm.paper_category, '') AS paper_category,
                      COALESCE(pm.reading_status, '') AS reading_status,
                      COALESCE(pm.note, '') AS note,
                      COALESCE(pm.suggested_category, '') AS suggested_category,
                      COALESCE(pm.dismissed_category, '') AS dismissed_category
                    FROM pdf_files pf
                    LEFT JOIN paper_meta pm ON pm.sha1 = pf.sha1
                    WHERE pf.sha1 = ?
                    """,
                    (sha1_clean,),
                ).fetchone()
                if not row:
                    continue

                user_tags = [
                    str(tag_row["tag_label"] or "").strip()
                    for tag_row in conn.execute(
                        """
                        SELECT tag_label
                        FROM paper_tags
                        WHERE sha1 = ? AND source = 'user'
                        ORDER BY LOWER(tag_label), tag_label
                        """,
                        (sha1_clean,),
                    ).fetchall()
                    if str(tag_row["tag_label"] or "").strip()
                ]
                dismissed_tags = [
                    str(tag_row["tag_label"] or "").strip()
                    for tag_row in conn.execute(
                        """
                        SELECT tag_label
                        FROM paper_tags
                        WHERE sha1 = ? AND source = 'dismissed'
                        ORDER BY LOWER(tag_label), tag_label
                        """,
                        (sha1_clean,),
                    ).fetchall()
                    if str(tag_row["tag_label"] or "").strip()
                ]

                heuristic_category, heuristic_tags = self._generate_suggestions_for_row(
                    row=row,
                    current_user_tags=user_tags,
                    dismissed_tags=dismissed_tags,
                    taxonomy_stats=taxonomy_stats,
                    dismissed_category=str(row["dismissed_category"] or ""),
                )
                llm_category = ""
                llm_tags: list[str] = []
                if llm_client is not None:
                    llm_category, llm_tags = self._generate_llm_suggestions_for_row(
                        llm_client=llm_client,
                        row=row,
                        current_user_tags=user_tags,
                        dismissed_tags=dismissed_tags,
                        taxonomy_stats=taxonomy_stats,
                        dismissed_category=str(row["dismissed_category"] or ""),
                        heuristic_category=heuristic_category,
                        heuristic_tags=heuristic_tags,
                    )
                suggested_category = llm_category or heuristic_category
                suggested_tags = self._merge_tag_suggestions(
                    primary=llm_tags,
                    secondary=heuristic_tags,
                    current_user_tags=user_tags,
                    dismissed_tags=dismissed_tags,
                )

                conn.execute(
                    """
                    INSERT INTO paper_meta (sha1, suggested_category, suggestion_updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(sha1) DO UPDATE SET
                      suggested_category = excluded.suggested_category,
                      suggestion_updated_at = excluded.suggestion_updated_at
                    """,
                    (sha1_clean, suggested_category, now),
                )
                conn.execute("DELETE FROM paper_tags WHERE sha1 = ? AND source = 'suggested'", (sha1_clean,))
                for tag in suggested_tags:
                    norm, label = self._normalize_tag(tag)
                    if not norm or not label:
                        continue
                    conn.execute(
                        """
                        INSERT INTO paper_tags (sha1, tag_norm, tag_label, source, updated_at)
                        VALUES (?, ?, ?, 'suggested', ?)
                        """,
                        (sha1_clean, norm, label, now),
                    )
                conn.commit()
                updated = self.get_paper_user_meta(sha1=sha1_clean)
                if updated:
                    updated_items.append(updated)

        return updated_items

    def apply_paper_suggestion_actions(
        self,
        *,
        sha1: str = "",
        path: Path | str | None = None,
        category_action: str = "",
        accept_tags: Iterable[str] | None = None,
        dismiss_tags: Iterable[str] | None = None,
        accept_all_tags: bool = False,
        dismiss_all_tags: bool = False,
    ) -> dict | None:
        current = self.get_paper_user_meta(sha1=sha1, path=path)
        if not current:
            return None

        current_user_tags = list(current.get("user_tags") or [])
        current_suggested_tags = list(current.get("suggested_tags") or [])
        current_dismissed_tags = list(current.get("dismissed_tags") or [])
        current_category = str(current.get("paper_category") or "")
        current_suggested_category = str(current.get("suggested_category") or "")
        current_dismissed_category = str(current.get("dismissed_category") or "")

        suggested_tag_map = {
            self._normalize_tag(tag)[0]: self._normalize_tag(tag)[1]
            for tag in current_suggested_tags
            if self._normalize_tag(tag)[0]
        }
        dismissed_tag_map = {
            self._normalize_tag(tag)[0]: self._normalize_tag(tag)[1]
            for tag in current_dismissed_tags
            if self._normalize_tag(tag)[0]
        }
        user_tag_map = {
            self._normalize_tag(tag)[0]: self._normalize_tag(tag)[1]
            for tag in current_user_tags
            if self._normalize_tag(tag)[0]
        }

        accept_norms = {
            norm
            for norm, _label in (self._normalize_tag(tag) for tag in (accept_tags or []))
            if norm and norm in suggested_tag_map
        }
        dismiss_norms = {
            norm
            for norm, _label in (self._normalize_tag(tag) for tag in (dismiss_tags or []))
            if norm and norm in suggested_tag_map
        }
        if accept_all_tags:
            accept_norms.update(suggested_tag_map.keys())
        if dismiss_all_tags:
            dismiss_norms.update(suggested_tag_map.keys())
        dismiss_norms.difference_update(accept_norms)

        next_user_tags = list(current_user_tags)
        next_user_seen = {self._normalize_tag(tag)[0] for tag in next_user_tags if self._normalize_tag(tag)[0]}
        for norm in accept_norms:
            label = suggested_tag_map.get(norm)
            if not label or norm in next_user_seen:
                continue
            next_user_seen.add(norm)
            next_user_tags.append(label)

        next_suggested_tags = [
            label
            for norm, label in suggested_tag_map.items()
            if norm not in accept_norms and norm not in dismiss_norms
        ]
        next_dismissed_tags = {
            norm: label
            for norm, label in dismissed_tag_map.items()
            if norm not in accept_norms
        }
        for norm in dismiss_norms:
            label = suggested_tag_map.get(norm)
            if label:
                next_dismissed_tags[norm] = label

        next_category = current_category
        next_suggested_category = current_suggested_category
        next_dismissed_category = current_dismissed_category
        if category_action == "accept" and current_suggested_category:
            next_category = current_suggested_category
            next_suggested_category = ""
            if next_dismissed_category == next_category:
                next_dismissed_category = ""
        elif category_action == "dismiss" and current_suggested_category:
            next_dismissed_category = current_suggested_category
            next_suggested_category = ""

        updated = self.upsert_paper_user_meta(
            sha1=str(current.get("sha1") or ""),
            paper_category=next_category,
            reading_status=str(current.get("reading_status") or ""),
            note=str(current.get("note") or ""),
            user_tags=next_user_tags,
        )
        if not updated:
            return None

        sha1_clean = str(updated.get("sha1") or "").strip().lower()
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE paper_meta
                SET suggested_category = ?, dismissed_category = ?, suggestion_updated_at = ?
                WHERE sha1 = ?
                """,
                (next_suggested_category, next_dismissed_category, now, sha1_clean),
            )
            conn.execute("DELETE FROM paper_tags WHERE sha1 = ? AND source IN ('suggested', 'dismissed')", (sha1_clean,))
            for tag in next_suggested_tags:
                norm, label = self._normalize_tag(tag)
                if not norm or not label:
                    continue
                conn.execute(
                    """
                    INSERT INTO paper_tags (sha1, tag_norm, tag_label, source, updated_at)
                    VALUES (?, ?, ?, 'suggested', ?)
                    """,
                    (sha1_clean, norm, label, now),
                )
            for tag in next_dismissed_tags.values():
                norm, label = self._normalize_tag(tag)
                if not norm or not label:
                    continue
                conn.execute(
                    """
                    INSERT INTO paper_tags (sha1, tag_norm, tag_label, source, updated_at)
                    VALUES (?, ?, ?, 'dismissed', ?)
                    """,
                    (sha1_clean, norm, label, now),
                )
        return self.get_paper_user_meta(sha1=sha1_clean)

    def batch_update_paper_user_meta(
        self,
        *,
        sha1s: Iterable[str] | None = None,
        paths: Iterable[Path | str] | None = None,
        apply_paper_category: bool = False,
        paper_category: str = "",
        apply_reading_status: bool = False,
        reading_status: str = "",
        add_tags: Iterable[str] | None = None,
        remove_tags: Iterable[str] | None = None,
    ) -> list[dict]:
        targets: list[str] = []
        seen: set[str] = set()

        for raw_sha1 in sha1s or []:
            sha1_clean = str(raw_sha1 or "").strip().lower()
            if not sha1_clean or sha1_clean in seen:
                continue
            seen.add(sha1_clean)
            targets.append(sha1_clean)

        for raw_path in paths or []:
            rec = self.ensure_record_for_path(raw_path)
            sha1_clean = str((rec or {}).get("sha1") or "").strip().lower()
            if not sha1_clean or sha1_clean in seen:
                continue
            seen.add(sha1_clean)
            targets.append(sha1_clean)

        if not targets:
            return []

        category_clean = self._clean_category(paper_category) if apply_paper_category else None
        status_clean = self._clean_reading_status(reading_status) if apply_reading_status else None
        add_tags_clean = self._clean_tags(add_tags)
        remove_tag_norms = {
            norm
            for norm, _label in (self._normalize_tag(tag) for tag in (remove_tags or []))
            if norm
        }

        updated_items: list[dict] = []
        for sha1_clean in targets:
            current = self.get_paper_user_meta(sha1=sha1_clean)
            if not current:
                continue

            merged_tags: list[str] = []
            merged_seen: set[str] = set()
            for raw_tag in list(current.get("user_tags") or []):
                norm, label = self._normalize_tag(raw_tag)
                if (not norm) or (norm in remove_tag_norms) or (norm in merged_seen):
                    continue
                merged_seen.add(norm)
                merged_tags.append(label)

            for raw_tag in add_tags_clean:
                norm, label = self._normalize_tag(raw_tag)
                if (not norm) or (norm in remove_tag_norms) or (norm in merged_seen):
                    continue
                merged_seen.add(norm)
                merged_tags.append(label)

            updated = self.upsert_paper_user_meta(
                sha1=sha1_clean,
                paper_category=category_clean if apply_paper_category and category_clean is not None else str(current.get("paper_category") or ""),
                reading_status=status_clean if apply_reading_status and status_clean is not None else str(current.get("reading_status") or ""),
                note=str(current.get("note") or ""),
                user_tags=merged_tags,
            )
            if updated:
                updated_items.append(updated)
        return updated_items

    def update_path(self, old_path: Path, new_path: Path) -> int:
        """
        Best-effort path update when a PDF file is renamed/moved on disk.
        Returns affected rows count.
        """
        old_s = str(Path(old_path))
        new_s = str(Path(new_path))
        with self._connect() as conn:
            cur = conn.execute("UPDATE pdf_files SET path = ? WHERE path = ?", (new_s, old_s))
            return int(getattr(cur, "rowcount", 0) or 0)

    def delete_by_path(self, path: Path) -> int:
        """
        Best-effort removal when a PDF file is deleted on disk.
        Returns affected rows count.
        """
        path_s = str(Path(path))
        with self._connect() as conn:
            row = conn.execute("SELECT sha1 FROM pdf_files WHERE path = ?", (path_s,)).fetchone()
            cur = conn.execute("DELETE FROM pdf_files WHERE path = ?", (path_s,))
            sha1 = str(row["sha1"] or "").strip().lower() if row and row["sha1"] else ""
            if sha1:
                conn.execute("DELETE FROM paper_tags WHERE sha1 = ?", (sha1,))
                conn.execute("DELETE FROM paper_meta WHERE sha1 = ?", (sha1,))
            return int(getattr(cur, "rowcount", 0) or 0)
