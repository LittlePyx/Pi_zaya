from __future__ import annotations

import time
from pathlib import Path
from typing import Optional


def _restore_formula_placeholders_if_needed(
    converter,
    md: Optional[str],
    formula_placeholders: Optional[dict[str, str]],
) -> Optional[str]:
    if md and formula_placeholders:
        return converter._restore_formula_placeholders(md, formula_placeholders)
    return md


def _call_llm_page_to_markdown(
    converter,
    *,
    png_bytes: bytes,
    page_index: int,
    total_pages: int,
    page_hint: str,
    speed_mode: str,
    is_references_page: bool,
    max_tokens_override: Optional[int] = None,
    formula_placeholders: Optional[dict[str, str]] = None,
) -> Optional[str]:
    call_kwargs = {
        "page_number": page_index,
        "total_pages": total_pages,
        "hint": page_hint,
        "speed_mode": speed_mode,
        "is_references_page": is_references_page,
    }
    if max_tokens_override is not None:
        call_kwargs["max_tokens_override"] = max_tokens_override
    md = converter.llm_worker.call_llm_page_to_markdown(
        png_bytes,
        **call_kwargs,
    )
    return _restore_formula_placeholders_if_needed(converter, md, formula_placeholders)


def _fallback_to_extraction_pipeline(
    converter,
    *,
    page,
    page_index: int,
    pdf_path: Path,
    assets_dir: Path,
    reason: str,
) -> Optional[str]:
    try:
        return converter._process_page(
            page,
            page_index=page_index,
            pdf_path=pdf_path,
            assets_dir=assets_dir,
        )
    except Exception as e:
        print(
            f"[VISION_DIRECT] {reason} on page {page_index+1}: {e}",
            flush=True,
        )
        return None


def _retry_empty_vision_output(
    converter,
    *,
    png_bytes: bytes,
    page_index: int,
    total_pages: int,
    page_hint: str,
    speed_mode: str,
    is_references_page: bool,
    max_tokens_override: Optional[int] = None,
    formula_placeholders: Optional[dict[str, str]] = None,
) -> Optional[str]:
    last_vl_err = ""
    try:
        last_vl_err = str(converter.llm_worker.get_last_vl_error_code() or "").strip().lower()
    except Exception:
        last_vl_err = ""

    if last_vl_err == "timeout":
        print(
            f"[VISION_DIRECT] VL hard-timeout on page {page_index+1}, skip empty retries and fallback",
            flush=True,
        )
        retry_n = 0
    elif last_vl_err == "unsupported_vision":
        print(
            f"[VISION_DIRECT] provider/model does not support image payloads on page {page_index+1}, skip empty retries and fallback",
            flush=True,
        )
        retry_n = 0
    else:
        retry_n = converter._vision_empty_retry_attempts()

    retry_sleep = converter._vision_empty_retry_backoff_s()
    for k in range(1, retry_n + 1):
        try:
            if retry_sleep > 0:
                time.sleep(retry_sleep)
        except Exception:
            pass

        print(
            f"[VISION_DIRECT] VL empty on page {page_index+1}, retry {k}/{retry_n}",
            flush=True,
        )
        retry_hint = (
            (page_hint + " " if page_hint else "")
            + "Previous attempt returned empty. OCR the full page and return complete Markdown only."
        )
        md = _call_llm_page_to_markdown(
            converter,
            png_bytes=png_bytes,
            page_index=page_index,
            total_pages=total_pages,
            page_hint=retry_hint,
            speed_mode=speed_mode,
            is_references_page=is_references_page,
            max_tokens_override=max_tokens_override,
            formula_placeholders=formula_placeholders,
        )
        if md:
            return md

    return None


def convert_page_with_vision_guardrails(
    converter,
    *,
    png_bytes: bytes,
    page,
    page_index: int,
    total_pages: int,
    page_hint: str,
    speed_mode: str,
    is_references_page: bool,
    pdf_path: Path,
    assets_dir: Path,
    image_names: Optional[list[str]] = None,
    max_tokens_override: Optional[int] = None,
    formula_placeholders: Optional[dict[str, str]] = None,
    skip_references_column_mode: bool = False,
) -> Optional[str]:
    """
    Run vision-direct page OCR with a math quality gate:
    1) normal VL page conversion
    2) retry once with stricter hint if math is fragmented
    3) fallback to block pipeline when fragmentation persists
    """
    if (
        is_references_page
        and (not skip_references_column_mode)
        and converter._vision_references_column_mode_enabled()
    ):
        try:
            md_ref = converter._convert_references_page_with_column_vl(
                page=page,
                page_index=page_index,
                total_pages=total_pages,
                page_hint=page_hint,
                speed_mode=speed_mode,
            )
            if md_ref:
                return md_ref
        except Exception as e:
            print(
                f"[VISION_DIRECT] references column OCR failed on page {page_index+1}: {e}",
                flush=True,
            )

    if not is_references_page:
        try:
            md_layout = converter._convert_page_with_layout_crops(
                page=page,
                page_index=page_index,
                total_pages=total_pages,
                page_hint=page_hint,
                speed_mode=speed_mode,
                pdf_path=pdf_path,
                assets_dir=assets_dir,
                image_names=image_names or [],
            )
            if md_layout:
                return md_layout
        except Exception as e:
            print(
                f"[VISION_DIRECT] structured layout OCR failed on page {page_index+1}: {e}",
                flush=True,
            )

    md = _call_llm_page_to_markdown(
        converter,
        png_bytes=png_bytes,
        page_index=page_index,
        total_pages=total_pages,
        page_hint=page_hint,
        speed_mode=speed_mode,
        is_references_page=is_references_page,
        max_tokens_override=max_tokens_override,
        formula_placeholders=formula_placeholders,
    )
    if not md:
        md = _retry_empty_vision_output(
            converter,
            png_bytes=png_bytes,
            page_index=page_index,
            total_pages=total_pages,
            page_hint=page_hint,
            speed_mode=speed_mode,
            is_references_page=is_references_page,
            max_tokens_override=max_tokens_override,
            formula_placeholders=formula_placeholders,
        )

    if not md:
        print(
            f"[VISION_DIRECT] VL returned empty for page {page_index+1}, falling back to extraction pipeline",
            flush=True,
        )
        return _fallback_to_extraction_pipeline(
            converter,
            page=page,
            page_index=page_index,
            pdf_path=pdf_path,
            assets_dir=assets_dir,
            reason="fallback extraction failed",
        )

    # References pages should not contain math; skip fragmented-math checks there.
    if is_references_page or not converter._vision_math_quality_gate_enabled():
        return md

    if not converter._looks_fragmented_math_output(md):
        return md

    print(
        f"[VISION_DIRECT] fragmented math detected on page {page_index+1}, retrying with strict formula hint",
        flush=True,
    )
    md_retry = _call_llm_page_to_markdown(
        converter,
        png_bytes=png_bytes,
        page_index=page_index,
        total_pages=total_pages,
        page_hint=converter._vision_math_retry_hint(page_hint),
        speed_mode=speed_mode,
        is_references_page=is_references_page,
        max_tokens_override=max_tokens_override,
        formula_placeholders=formula_placeholders,
    )
    if md_retry and (not converter._looks_fragmented_math_output(md_retry)):
        print(
            f"[VISION_DIRECT] page {page_index+1} math recovered after retry",
            flush=True,
        )
        return md_retry

    if not converter._vision_fragment_fallback_enabled():
        print(
            f"[VISION_DIRECT] fragmented math persists on page {page_index+1}, keep VL output (fallback disabled)",
            flush=True,
        )
        # Keep VL output by default; block-level fallback can alter heading/layout style.
        return md_retry or md

    print(
        f"[VISION_DIRECT] fragmented math persists on page {page_index+1}, using extraction fallback",
        flush=True,
    )
    fallback_md = _fallback_to_extraction_pipeline(
        converter,
        page=page,
        page_index=page_index,
        pdf_path=pdf_path,
        assets_dir=assets_dir,
        reason="fallback extraction failed",
    )
    if fallback_md:
        return fallback_md

    # Last resort: keep the best VL output we have rather than dropping the page.
    return md_retry or md
