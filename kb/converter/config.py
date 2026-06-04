from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LlmConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    request_sleep_s: float = 0.0
    timeout_s: float = 45.0
    max_retries: int = 0


@dataclass(frozen=True)
class ConvertConfig:
    pdf_path: Path
    out_dir: Path
    translate_zh: bool
    start_page: int
    end_page: int
    skip_existing: bool
    keep_debug: bool
    llm: Optional[LlmConfig]
    llm_classify: bool = True
    llm_render_page: bool = False
    llm_classify_only_if_needed: bool = True
    classify_batch_size: int = 40
    image_scale: float = 2.2
    figure_dpi: int | None = 320
    image_alpha: bool = False
    detect_tables: bool = True
    table_pdfplumber_fallback: bool = False
    eq_image_fallback: bool = False
    global_noise_scan: bool = True
    llm_repair: bool = True
    llm_repair_body_math: bool = False
    llm_smart_math_repair: bool = True
    llm_auto_page_render_threshold: int = 12
    llm_workers: int = 1
    workers: int = 1
    speed_mode: str = "normal"  # "normal" (vision-direct with max parallelism), "ultra_fast" (faster, lower quality), "no_llm" (basic text extraction)


@dataclass(frozen=True)
class TextBlock:
    bbox: tuple[float, float, float, float]
    text: str
    max_font_size: float
    is_bold: bool
    insert_image: Optional[str] = None
    is_code: bool = False
    is_table: bool = False
    table_markdown: Optional[str] = None
    is_math: bool = False
    heading_level: Optional[int] = None
