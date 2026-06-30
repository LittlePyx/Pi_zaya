from __future__ import annotations

from dataclasses import replace
import os
import re
from typing import Callable

from api.reference_summary_text import _clean_summary_line, _has_cjk_text, _has_latin_text, _summary_excerpt


def _llm_summarize_abstract_zh(
    title: str,
    abstract_text: str,
    *,
    load_settings_func: Callable[[], object],
    chat_cls: Callable[[object], object],
    is_summary_quality_ok: Callable[[str], bool],
) -> str:
    abs_text = _summary_excerpt(abstract_text, max_sentences=5, max_len=900)
    title_text = _clean_summary_line(title)
    if not abs_text:
        return ""
    raw_flag = str(os.environ.get("KB_CITE_SUMMARY_USE_LLM", "0") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
        return ""
    try:
        settings = load_settings_func()
    except Exception:
        return ""
    if not getattr(settings, "api_key", None):
        return ""
    try:
        fast_settings = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 20.0),
            max_retries=1,
        )
    except Exception:
        fast_settings = settings
    try:
        ds = chat_cls(fast_settings)
        out = (
            ds.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是科研论文助手。请基于给定信息输出2-3句中文学术概括，要求："
                            "第1句说明研究问题或目标；"
                            "第2句说明核心方法或机制（作者具体做了什么）；"
                            "第3句说明关键结果、贡献或适用边界（若摘要未给量化指标需明确说明）。"
                            "严禁编造数据或结论，严禁只复述标题。只输出概括正文。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"论文标题：{title_text}\n"
                            f"摘要原文：{abs_text}\n\n"
                            "请给出中文学术概括："
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=360,
            )
            or ""
        ).strip()
    except Exception:
        return ""
    out = _summary_excerpt(out, max_sentences=3, max_len=360)
    if not _has_cjk_text(out):
        return ""
    if not is_summary_quality_ok(out):
        return ""
    return out


def _translate_summary_to_zh(
    text: str,
    *,
    load_settings_func: Callable[[], object],
    chat_cls: Callable[[object], object],
) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    src = _summary_excerpt(src, max_sentences=3, max_len=520)
    if not src:
        return ""
    if _has_cjk_text(src) and (not _has_latin_text(src)):
        return src
    raw_flag = str(os.environ.get("KB_CITE_SUMMARY_TRANSLATE_ZH", "1") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
        return src
    try:
        settings = load_settings_func()
    except Exception:
        return src
    if not getattr(settings, "api_key", None):
        return src
    try:
        fast_settings = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 8.0),
            max_retries=0,
        )
    except Exception:
        fast_settings = settings
    try:
        ds = chat_cls(fast_settings)
        out = (
            ds.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "将给定文献摘要改写为中文学术概括，输出 2-3 句。"
                            "要求："
                            "1) 尽量覆盖研究问题/方法/主要结果或贡献；"
                            "2) 术语准确、语气学术；"
                            "3) 不编造原文没有的信息；"
                            "4) 只输出概括正文，不要列表或前缀标签。"
                        ),
                    },
                    {"role": "user", "content": src},
                ],
                temperature=0.0,
                max_tokens=320,
            )
            or ""
        ).strip()
    except Exception:
        return src
    out = re.sub(r"\s+", " ", out).strip()
    if not out:
        return src
    if not _has_cjk_text(out):
        return src
    return _summary_excerpt(out, max_sentences=3, max_len=360)


def _finalize_abstract_summary_line(
    *,
    title: str,
    abstract_text: str,
    llm_summarize_abstract_zh: Callable[..., str],
    translate_summary_to_zh: Callable[[str], str],
) -> tuple[str, str]:
    abstract_line = _summary_excerpt(abstract_text, max_sentences=5, max_len=900)
    if not abstract_line:
        return "", ""
    llm_summary = llm_summarize_abstract_zh(title=title, abstract_text=abstract_line)
    if llm_summary:
        return llm_summary, "llm_abstract"
    translated = translate_summary_to_zh(abstract_line)
    if translated:
        return translated, "translated_abstract"
    return abstract_line, "translated_abstract"
