from __future__ import annotations

import os
import json
import re
import time
import base64
import threading
from typing import Optional, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .config import ConvertConfig
from .models import TextBlock
from .text_utils import _normalize_text
from .tables import _is_markdown_table_sane
from .post_processing import (
    fix_math_markdown,
    _normalize_math_for_typora,
    _fix_malformed_code_fences,
)

class LLMWorker:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg
        self._client = None
        # Global-ish concurrency limiter: one LLMWorker is shared across page threads in PDFConverter.
        # This prevents "N pages in parallel" from flooding the provider and stalling on throttling.
        self._llm_sem: threading.Semaphore | None = None
        self._llm_max_inflight: int = 8
        try:
            raw = str(os.environ.get("KB_LLM_MAX_INFLIGHT", "") or "").strip()
            if raw:
                max_inflight = int(raw)
            else:
                # Keep a stable default for full-page VL OCR.
                # Too high concurrency commonly triggers provider-side timeout/rate-limit cascades.
                max_inflight = 8
            max_inflight = max(1, min(32, int(max_inflight)))
            self._llm_max_inflight = int(max_inflight)
            self._llm_sem = threading.Semaphore(max_inflight)
        except Exception:
            self._llm_max_inflight = 8
            self._llm_sem = threading.Semaphore(self._llm_max_inflight)
        # Small in-memory caches to avoid repeated calls for identical snippets.
        # These caches live per Streamlit process and reset on restart.
        self._cache_confirm_heading: dict[str, dict] = {}
        self._cache_repair_math: dict[str, str] = {}
        self._cache_max_items = 2048
        if self.cfg.llm:
            try:
                self._client = self._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"[WARN] Failed to init OpenAI client: {e}")
                self._client = None

    def _ensure_openai_class(self):
        if OpenAI is None:
            raise ImportError("openai module not installed.")
        return OpenAI

    def get_llm_max_inflight(self) -> int:
        try:
            return max(1, int(self._llm_max_inflight))
        except Exception:
            return 8

    @staticmethod
    def _messages_contain_image_payload(messages: Any) -> bool:
        try:
            if not isinstance(messages, list):
                return False
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if isinstance(item, dict) and str(item.get("type", "")).strip().lower() == "image_url":
                        return True
            return False
        except Exception:
            return False

    def _llm_create(self, **kwargs):
        if not self._client:
            raise RuntimeError("LLM client not initialized")
        # Keep call-timeouts and retries configurable; defaults are conservative.
        timeout_s = 45.0
        max_retries = 0
        try:
            if self.cfg.llm:
                timeout_s = float(getattr(self.cfg.llm, "timeout_s", timeout_s) or timeout_s)
                max_retries = int(getattr(self.cfg.llm, "max_retries", max_retries) or max_retries)
        except Exception:
            timeout_s = 45.0
            max_retries = 0

        # Vision page OCR is heavier than text-only calls; keep a safer timeout floor.
        try:
            if self._messages_contain_image_payload(kwargs.get("messages")):
                raw_v_to = str(os.environ.get("KB_PDF_VISION_TIMEOUT_S", "120") or "120").strip()
                try:
                    vision_timeout_floor = float(raw_v_to)
                except Exception:
                    vision_timeout_floor = 120.0
                timeout_s = max(float(timeout_s), max(30.0, min(300.0, vision_timeout_floor)))
        except Exception:
            pass

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if self._llm_sem is None:
                    return self._client.chat.completions.create(
                        model=self.cfg.llm.model,
                        timeout=timeout_s,
                        **kwargs,
                    )
                # Increase semaphore acquire timeout for vision-direct mode (full-page screenshots take longer)
                # Default: wait up to 60 seconds for a slot (was 15s), or 2x the request timeout, whichever is larger
                sem_timeout = max(60.0, float(timeout_s) * 2.0)
                # But cap at 120s to avoid infinite waits
                sem_timeout = min(120.0, sem_timeout)
                acquired = self._llm_sem.acquire(timeout=sem_timeout)
                if not acquired:
                    raise TimeoutError(
                        f"LLM inflight slots saturated (KB_LLM_MAX_INFLIGHT). "
                        f"Waited {sem_timeout:.1f}s for a slot. Consider increasing KB_LLM_MAX_INFLIGHT."
                    )
                try:
                    return self._client.chat.completions.create(
                        model=self.cfg.llm.model,
                        timeout=timeout_s,
                        **kwargs,
                    )
                finally:
                    try:
                        self._llm_sem.release()
                    except Exception:
                        pass
            except TimeoutError as e:
                # For semaphore timeout, retry with backoff (up to max_retries)
                last_err = e
                if attempt < max_retries:
                    # Exponential backoff: 2s, 4s, 8s...
                    backoff = min(10.0, 2.0 * (2**attempt))
                    time.sleep(backoff)
                    continue
                # If all retries exhausted, raise the timeout
                raise
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    break
                # Short exponential backoff; keeps UI responsive.
                time.sleep(min(1.2, 0.25 * (2**attempt)))
        assert last_err is not None
        raise last_err

    def _get_max_tokens_for_vision(
        self,
        speed_mode: str = 'normal',
        *,
        is_references_page: bool = False,
    ) -> int:
        """Get max_tokens for vision calls, with an optional tighter cap for references pages."""
        try:
            raw = str(os.environ.get("KB_PDF_VISION_MAX_TOKENS", "") or "").strip()
            if raw:
                return max(1024, min(8192, int(raw)))  # Clamp between 1024-8192
        except Exception:
            pass
        # Default based on speed mode
        defaults = {
            'normal': 3072,
            'ultra_fast': 2048,
            'no_llm': 0,
        }
        default = defaults.get(speed_mode, 3072)
        if is_references_page:
            try:
                raw_ref = str(os.environ.get("KB_PDF_VISION_REFS_MAX_TOKENS", "") or "").strip()
                if raw_ref:
                    default = max(1024, min(4096, int(raw_ref)))
                else:
                    # References OCR is text-dense; a lower cap cuts tail latency while preserving content.
                    default = min(default, 2048)
            except Exception:
                default = min(default, 2048)
        config_val = int(getattr(self.cfg.llm, "max_tokens", 0) or 0)
        if config_val > 0:
            if is_references_page:
                return min(config_val, 3072)
            return min(config_val, 4096)  # Respect config but cap at 4096
        return default

    def _sanitize_vl_markdown(self, md: str, *, is_references_page: bool = False) -> str:
        """
        Deterministic safety pass for VL page output.
        - Cleanup false-math wrappers and split/garbled display math.
        - For references pages, force plain-text citations (no math delimiters).
        """
        def _is_placeholder_line(s: str) -> bool:
            t = (s or "").strip().lower()
            if not t:
                return False
            # Common VL/OCR placeholders when text is clipped or low-confidence.
            if "(incomplete visible)" in t:
                return True
            if "(partially visible)" in t:
                return True
            if "(not fully visible)" in t:
                return True
            if re.search(r"\b(?:incomplete|illegible|unreadable)\s+visible\b", t):
                return True
            if "[unreadable]" in t or "[illegible]" in t:
                return True
            return False

        out = (md or "").strip()
        if not out:
            return ""

        try:
            out = fix_math_markdown(out)
        except Exception:
            pass
        try:
            out = _normalize_math_for_typora(out)
        except Exception:
            pass
        # Repair malformed fenced code blocks early at page level so one bad fence
        # cannot swallow all subsequent content in merged markdown.
        try:
            out = _fix_malformed_code_fences(out)
        except Exception:
            pass

        if is_references_page:
            lines = out.splitlines()
            cleaned: list[str] = []
            in_display_math = False
            for ln in lines:
                st = ln.strip()
                if st == "$$":
                    in_display_math = not in_display_math
                    continue
                if in_display_math:
                    ln = st
                # Remove inline math wrappers, keep content.
                ln = re.sub(r"\$([^$\n]{1,400})\$", r"\1", ln)
                ln = ln.replace("$$", "").replace("$", "")
                if _is_placeholder_line(ln):
                    continue
                cleaned.append(ln)
            out = "\n".join(cleaned)
        else:
            # Generic cleanup: drop explicit OCR placeholder lines that should
            # never appear in final markdown content.
            out = "\n".join(ln for ln in out.splitlines() if not _is_placeholder_line(ln))

        return out.strip()

    def _cache_set(self, cache: dict, key: str, val) -> None:
        cache[key] = val
        # Simple size bound (drop oldest insertion order in Py>=3.7 dict).
        if len(cache) > int(self._cache_max_items):
            try:
                for k in list(cache.keys())[: max(1, len(cache) // 3)]:
                    cache.pop(k, None)
            except Exception:
                cache.clear()

    def _extract_json_array(self, s: str) -> Optional[list]:
        if not s:
            return None
        start = s.find("[")
        end = s.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        blob = s[start : end + 1]
        try:
            return json.loads(blob)
        except Exception:
            return None

    def call_llm_repair_table(self, raw_table: str, *, page_number: int, block_index: int) -> Optional[str]:
        if not self.cfg.llm:
            return None
        text = raw_table.strip()
        if not text:
            return None
        
        prompt = (
            f"Fix this broken text table from PDF page {page_number+1} into a clean Markdown table.\n"
            "Return ONLY the Markdown table. No other text.\n"
            "If it is definitely not a table, return 'NOT_A_TABLE'.\n\n"
            f"RAW TEXT:\n{text}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a precise table fixer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
        
        out = (resp.choices[0].message.content or "").strip()
        if out == "NOT_A_TABLE":
            return None
        if "```" in out:
            # Extract content from code block
            m = re.search(r"```(?:\w+)?\n(.*?)```", out, re.DOTALL)
            if m:
                out = m.group(1).strip()
        return out

    def call_llm_repair_math(
        self,
        raw_math: str,
        *,
        page_number: int,
        block_index: int,
        context_before: str = "",
        context_after: str = "",
        eq_number: Optional[str] = None
    ) -> Optional[str]:
        if not self.cfg.llm:
            return None

        cache_key = None
        try:
            norm = _normalize_text(raw_math or "").strip()
            if norm:
                cache_key = f"math:{page_number}:{block_index}:{eq_number or ''}:{norm[:800]}"
                cached = self._cache_repair_math.get(cache_key)
                if isinstance(cached, str) and cached.strip():
                    return cached
        except Exception:
            cache_key = None
        
        ctx_prompt = ""
        if context_before:
            ctx_prompt += f"\nContext before:\n...{context_before[-300:]}\n"
        if context_after:
            ctx_prompt += f"\nContext after:\n{context_after[:300]}...\n"
            
        eq_hint = f"(Equation number: {eq_number})" if eq_number else ""
        prompt = (
            f"Recover this garbled math equation from PDF page {page_number+1} {eq_hint}.\n"
            "Return ONLY the LaTeX code (no $/$$ delimiters, no \\begin{equation}/align).\n"
            "Compatibility rules (strict):\n"
            "- Output Typora/KaTeX-compatible LaTeX only.\n"
            "- Do NOT use custom macros, \\newcommand, \\def, or \\DeclareMathOperator.\n"
            "- Use standard operators: e.g., \\operatorname*{arg\\,min}, \\operatorname*{arg\\,max}.\n"
            "CRITICAL fidelity rules:\n"
            "- Do NOT invent new variable names or symbols.\n"
            "- Preserve the original identifiers as much as possible (e.g., M vs A, C vs X).\n"
            "- Do NOT add explanatory prose.\n"
            "- If unsure about a piece, keep it minimal/faithful rather than guessing.\n"
            "If it's a display equation, return standard LaTeX for the equation body.\n"
            f"{ctx_prompt}\n"
            f"GARBLED BLOCK:\n{raw_math}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a LaTeX math expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
            
        out = (resp.choices[0].message.content or "").strip()
        # Remove markdown fences if the model ignored instructions.
        if out.startswith("```"):
            try:
                m = re.search(r"```(?:\w+)?\n(.*?)```", out, re.DOTALL)
                if m:
                    out = (m.group(1) or "").strip()
            except Exception:
                pass
        # Strip $$...$$ or \[...\]
        if out.startswith("$$") and out.endswith("$$"):
            out = out[2:-2].strip()
        elif out.startswith("\\[") and out.endswith("\\]"):
            out = out[2:-2].strip()

        # Validate: if the model returned an explanation / prose, treat as failure.
        # (This prevents huge paragraphs being wrapped into $$ ... $$.)
        try:
            out_s = out.strip()
            # Disallow equation environments; we only want raw LaTeX math content.
            if re.search(r"\\begin\{equation\}|\\end\{equation\}|\\begin\{align\}|\\end\{align\}", out_s):
                return None
            word_n = len(re.findall(r"\b\w+\b", out_s))
            letters_n = len(re.findall(r"[A-Za-z]", out_s))
            has_sentence = (". " in out_s) or ("? " in out_s) or ("! " in out_s)
            if (len(out_s) >= 160 and word_n >= 22 and letters_n >= 80 and has_sentence):
                return None
            # Common refusal/explanation patterns
            bad_markers = [
                "the garbled block",
                "based on the notation",
                "it likely represents",
                "interpretation",
                "here is the latex",
                "here is the laTeX",
            ]
            low = out_s.lower()
            if any(x in low for x in bad_markers):
                return None
        except Exception:
            pass

        if cache_key and out:
            try:
                self._cache_set(self._cache_repair_math, cache_key, out)
            except Exception:
                pass
        return out

    def call_llm_repair_math_from_image(
        self,
        png_bytes: bytes,
        *,
        page_number: int,
        block_index: int,
        eq_number: Optional[str] = None,
    ) -> Optional[str]:
        """
        Vision-based math recovery: read the equation image and output LaTeX.
        This is optional and only works if the configured model/backend supports image inputs.
        """
        if not self.cfg.llm or not self._client:
            return None
        if not png_bytes:
            return None

        # Cache by content hash to avoid repeated vision calls.
        cache_key = None
        try:
            import hashlib
            h = hashlib.sha1(png_bytes).hexdigest()[:20]
            cache_key = f"math_vision:{self.cfg.llm.model}:{page_number}:{block_index}:{eq_number or ''}:{h}"
            cached = self._cache_repair_math.get(cache_key)
            if isinstance(cached, str) and cached.strip():
                return cached
        except Exception:
            cache_key = None

        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        eq_hint = f"(Equation number: {eq_number})" if eq_number else ""

        prompt = (
            f"Recover the LaTeX for this equation image from PDF page {page_number+1} {eq_hint}.\n"
            "Return ONLY the LaTeX for the equation body.\n"
            "- No $/$$ delimiters\n"
            "- No \\begin{equation}/align environments\n"
            "- Typora/KaTeX-compatible LaTeX only (no custom macros)\n"
            "- No \\newcommand / \\def / \\DeclareMathOperator\n"
            "- No explanations\n"
            "Be exact and faithful to the image.\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        debug_vision = False
        try:
            debug_vision = bool(int(os.environ.get("KB_PDF_DEBUG_VISION_MATH", "0") or "0")) or bool(
                getattr(self.cfg, "keep_debug", False)
            )
        except Exception:
            debug_vision = False
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a LaTeX math expert. Return only LaTeX."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=min(900, int(getattr(self.cfg.llm, "max_tokens", 4096) or 4096)),
            )
        except Exception as e:
            if debug_vision:
                try:
                    m = str(getattr(self.cfg.llm, "model", "") or "")
                except Exception:
                    m = ""
                print(
                    f"[VISION_MATH] call failed: page={page_number+1} block={block_index+1} model={m!a} err={e!a}",
                    flush=True,
                )
            return None

        out = (resp.choices[0].message.content or "").strip()
        # Strip fences / delimiters if model disobeyed.
        if out.startswith("```"):
            try:
                m = re.search(r"```(?:\w+)?\n(.*?)```", out, re.DOTALL)
                if m:
                    out = (m.group(1) or "").strip()
            except Exception:
                pass
        if out.startswith("$$") and out.endswith("$$"):
            out = out[2:-2].strip()
        if re.search(r"\\begin\{equation\}|\\begin\{align\}", out):
            return None
        # Reject obvious non-LaTeX (tables / explanations) to avoid corrupting math blocks.
        try:
            out_s = out.strip()
            # Table-like output
            if re.search(r"(?m)^\s*\|.*\|\s*$", out_s) and out_s.count("|") >= 6:
                return None
            word_n = len(re.findall(r"\b\w+\b", out_s))
            letters_n = len(re.findall(r"[A-Za-z]", out_s))
            has_sentence = (". " in out_s) or ("? " in out_s) or ("! " in out_s)
            if (len(out_s) >= 180 and word_n >= 28 and letters_n >= 80 and has_sentence):
                return None
            bad_markers = ["where ", "denotes ", "this equation", "the equation", "we can see", "it represents"]
            low = out_s.lower()
            if any(x in low for x in bad_markers):
                return None
        except Exception:
            pass
        if cache_key and out:
            try:
                self._cache_set(self._cache_repair_math, cache_key, out)
            except Exception:
                pass
        return out or None

    def call_llm_page_to_markdown(
        self,
        png_bytes: bytes,
        *,
        page_number: int,
        total_pages: int = 0,
        hint: str = "",
        speed_mode: str = 'normal',
        is_references_page: bool = False,
    ) -> Optional[str]:
        """
        Vision-based full-page conversion: send a page screenshot to the VL model
        and get back Markdown directly.  This bypasses all text-extraction / block-
        classification logic and relies entirely on the vision model's OCR + layout
        understanding.
        """
        if not self.cfg.llm or not self._client:
            return None
        if not png_bytes:
            return None

        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        page_hint = f" (page {page_number + 1}"
        if total_pages > 0:
            page_hint += f" of {total_pages}"
        page_hint += ")"
        extra = f"\nAdditional context: {hint}" if hint else ""

        page_type_notice = ""
        if is_references_page:
            page_type_notice = (
                "**PAGE TYPE: REFERENCES/BIBLIOGRAPHY**\n"
                "- This page is references content.\n"
                "- Output plain-text references only.\n"
                "- One complete reference per line.\n"
                "- If an item is clipped/uncertain, skip it (do not output placeholders).\n"
                "- Do NOT use `$...$`, `$$...$$`, or code fences on this page.\n\n"
            )

        prompt = (
            f"Convert this PDF page image{page_hint} to Markdown.{extra}\n\n"
            f"{page_type_notice}"
            "Requirements (strict):\n"
            "1. Reproduce all visible body text faithfully. Do not summarize.\n"
            "2. Keep section hierarchy with Markdown headings: # / ## / ### / #### when appropriate.\n"
            "3. Exclude non-body metadata (journal headers/footers, websites, page counters like '(n of m)', copyright/publisher boilerplate, isolated affiliation/contact/ORCID/DOI footer blocks).\n"
            "4. Tables must be valid Markdown tables with all cells preserved.\n"
            "5. Keep figure/image references and captions when present.\n"
            "6. Use `$...$` or `$$...$$` ONLY for true mathematical expressions.\n"
            "7. NEVER wrap prose, citations (`[12]`), headings, names, references, or metadata in math delimiters.\n"
            "8. If one display equation is visually split across lines, reconstruct it into ONE coherent `$$...$$` block.\n"
            "9. If equation number `(N)` is visible next to a display equation, append `\\tag{N}`.\n"
            "10. LaTeX must be Typora/KaTeX-compatible. No custom macros (`\\newcommand`, `\\def`, `\\DeclareMathOperator`).\n"
            "11. For references pages: plain-text references only, one full reference per line, keep `[N]` numbering, no math delimiters/code fences.\n"
            "12. Never output placeholders like '(incomplete visible)', 'unreadable', 'illegible', or diagnostics.\n"
            "13. Return ONLY Markdown content. Do not output explanations, diagnostics, or refusal text.\n"
        )

        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)

        system_content = (
            "You are an expert document converter. "
            "You convert PDF page images into clean, faithful Markdown with correct LaTeX math. "
            "All LaTeX must be Typora/KaTeX-compatible and must not rely on custom macro definitions. "
            "Only mark true mathematical expressions with $...$ or $$...$$; never wrap prose/citations/metadata in math delimiters. "
            "Exclude non-body metadata (author affiliation/contact blocks, journal headers/footers, DOI-only footer lines, copyright/publisher boilerplate). "
            "Return ONLY the Markdown. No explanations."
        )
        if is_references_page:
            system_content = (
                "You are formatting an academic REFERENCES page. "
                "Output plain text references only. One reference per line. "
                "If an entry is cut off or uncertain, omit it instead of emitting placeholders. "
                "Never use $...$ or $$...$$ or code fences. "
                "Return ONLY Markdown plain-text lines."
            )

        try:
            resp = self._llm_create(
                messages=[
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=self._get_max_tokens_for_vision(
                    speed_mode=speed_mode,
                    is_references_page=is_references_page,
                ),
            )
        except Exception as e:
            error_str = str(e)
            error_msg = f"[VISION_PAGE] error page={page_number + 1} err={e!a}"
            
            # Check for specific API errors (ASCII-only logs for Windows GBK consoles).
            if "Access denied" in error_str or "account is in good standing" in error_str:
                print(f"{error_msg}", flush=True)
                print(
                    "[VISION_PAGE] API access denied. Check API key, account balance/status, and rate limits.",
                    flush=True,
                )
                print(
                    "[VISION_PAGE] Help: https://help.aliyun.com/zh/model-studio/error",
                    flush=True,
                )
            elif "400" in error_str or "BadRequestError" in error_str:
                print(f"{error_msg}", flush=True)
                if "image_url" in error_str and "expected `text`" in error_str:
                    print(
                        "[VISION_PAGE] API rejected image payload. The current model/provider endpoint likely does not support this vision message format.",
                        flush=True,
                    )
                else:
                    print(
                        "[VISION_PAGE] API bad request (400). Check model capability and request schema.",
                        flush=True,
                    )
            elif "401" in error_str or "Unauthorized" in error_str:
                print(f"{error_msg}", flush=True)
                print("[VISION_PAGE] API authentication failed (401). Check API key.", flush=True)
            elif "429" in error_str or "rate limit" in error_str.lower():
                print(f"{error_msg}", flush=True)
                print("[VISION_PAGE] API rate limited (429). Retry later.", flush=True)
            else:
                print(f"{error_msg}", flush=True)
            
            return None

        out = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if model wrapped the whole output
        if out.startswith("```markdown") or out.startswith("```md"):
            try:
                m = re.search(r"```(?:markdown|md)\n(.*?)```", out, re.DOTALL)
                if m:
                    out = (m.group(1) or "").strip()
            except Exception:
                pass
        elif out.startswith("```") and out.endswith("```"):
            out = out[3:]
            if out.endswith("```"):
                out = out[:-3]
            out = out.strip()
        try:
            out = self._sanitize_vl_markdown(out, is_references_page=is_references_page)
        except Exception:
            pass
        return out or None

    def call_llm_confirm_and_level_heading(
        self,
        heading_text: str,
        *,
        page_number: int,
        suggested_level: Optional[int] = None
    ) -> Optional[dict]:
        """Use LLM to confirm if text is a heading and determine its level.
        
        Returns:
            dict with keys: 'is_heading' (bool), 'level' (int), 'text' (str)
            or None if LLM unavailable
        """
        if not self.cfg.llm or not self._client:
            return None

        cache_key = None
        try:
            norm = _normalize_text(heading_text or "").strip()
            if norm:
                cache_key = f"heading:{suggested_level or ''}:{norm[:400]}"
                cached = self._cache_confirm_heading.get(cache_key)
                if isinstance(cached, dict) and ("is_heading" in cached):
                    return dict(cached)
        except Exception:
            cache_key = None
        
        prompt = f"""You are an expert at identifying research paper section headings. Analyze this text carefully.

Text: "{heading_text}"

CRITICAL: If the text matches ANY of these patterns, it is DEFINITELY a heading:
- Starts with a number followed by a dot and space: "1. ", "2. ", "3. ", etc.
- Starts with number.number: "3.1. ", "4.2. ", etc.
- Starts with number.number.number: "3.1.1. ", etc.
- Starts with a capital letter followed by dot and space: "A. ", "B. ", etc.
- Common section names: "Introduction", "Related Work", "Method", "Experiments", "Conclusion", "Abstract", "References"

Rules for heading levels:
- "1. Introduction", "2. Related Work", "3. Method", "4. Experiments", "5. Conclusion" → Level 1 (#) - main sections
- "3.1. Background", "3.2. Method", "4.1. Experimental Setup" → Level 2 (##) - subsections  
- "3.1.1. Details", "3.1.2. Implementation" → Level 3 (###) - sub-subsections
- "A. Appendix", "B. Proof" → Level 2 (##) - appendix sections
- "A.1. Details" → Level 3 (###) - appendix subsections

REJECT as heading ONLY if:
- It's clearly an author name (e.g., "Yunhao Li", "John Smith")
- It's clearly a university/affiliation (e.g., "Zhejiang University")
- It's a table header with metrics (e.g., "PSNR ↑ SSIM ↑")
- It's a pure math expression with no words (e.g., "x^2 + y^2")
- It's very short (≤3 chars) with no context

Return JSON with:
- "is_heading": true/false
- "level": 1/2/3 (only if is_heading is true, null otherwise)
- "text": cleaned heading text

Examples:
- "1. Introduction" → {{"is_heading": true, "level": 1, "text": "1. Introduction"}}
- "2. Related Work" → {{"is_heading": true, "level": 1, "text": "2. Related Work"}}
- "3. Method" → {{"is_heading": true, "level": 1, "text": "3. Method"}}
- "3.1. Background" → {{"is_heading": true, "level": 2, "text": "3.1. Background"}}
- "4.1. Experimental Setup" → {{"is_heading": true, "level": 2, "text": "4.1. Experimental Setup"}}
- "x^2 + y^2" → {{"is_heading": false, "level": null, "text": "x^2 + y^2"}}
- "Yunhao Li" → {{"is_heading": false, "level": null, "text": "Yunhao Li"}}

Return ONLY valid JSON, no other text:"""
        
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are an expert at identifying research paper section headings and determining their hierarchy. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,
            )
            result_text = (resp.choices[0].message.content or "").strip()
            # Remove markdown code fences if present
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:\w+)?\n?', '', result_text)
                result_text = re.sub(r'\n?```$', '', result_text)
            
            import json
            result = json.loads(result_text)
            if cache_key and isinstance(result, dict):
                try:
                    self._cache_set(self._cache_confirm_heading, cache_key, dict(result))
                except Exception:
                    pass
            return result
        except Exception as e:
            return None

    def call_llm_polish_code(self, raw_code: str, *, page_number: int, block_index: int) -> Optional[str]:
        if not self.cfg.llm:
            return None
            
        prompt = (
            f"Fix this OCR-damaged pseudocode/code from PDF page {page_number+1}.\n"
            "Preserve indentation. Fix arrow symbols (<- ->), assignment (:=), and keywords.\n"
            "Return ONLY the fixed code text. No markdown fences.\n\n"
            f"RAW CODE:\n{raw_code}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a code fixer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
        
        return (resp.choices[0].message.content or "").strip()

    def call_llm_repair_body_paragraph(self, text: str, *, page_number: int, block_index: int) -> Optional[str]:
        if not self.cfg.llm:
            return None
            
        prompt = (
            f"Fix this text paragraph from PDF page {page_number+1}.\n"
            "It may contain inline math that was OCR'd as garbage text.\n"
            "Convert inline math to LaTeX ($...$). Fix partial words.\n"
            "Return ONLY the fixed paragraph text.\n\n"
            f"RAW TEXT:\n{text}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a text fixer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
            
        return (resp.choices[0].message.content or "").strip()

    def call_llm_convert_page(self, text_content: str, *, page_num: int) -> str:
        """
        Full page conversion when heuristics fail.
        """
        if not self.cfg.llm:
            return text_content
            
        prompt = (
            f"Convert this raw PDF text from page {page_num+1} into clean, structured Markdown.\n"
            "Fix headers, lists, tables, and math.\n"
            "Return ONLY the Markdown.\n\n"
            f"RAW TEXT:\n{text_content}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a PDF to Markdown converter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm_render_max_tokens or self.cfg.llm.max_tokens,
            )
        except Exception:
            return text_content
            
        return (resp.choices[0].message.content or text_content).strip()

    def call_llm_classify_blocks(self, blocks: list[TextBlock], page_number: int, page_wh: tuple[float, float]) -> Optional[list[dict]]:
        if not self.cfg.llm or not self._client or not self.cfg.llm_classify:
            return None

        W, H = page_wh
        llm_cfg = self.cfg.llm

        def pack(bs: list[TextBlock], offset: int) -> list[dict]:
            out = []
            for i, b in enumerate(bs):
                txt = b.text.strip()
                if len(txt) > 800:
                    txt = txt[:800] + "..."
                out.append(
                    {
                        "i": offset + i,
                        "text": txt,
                        "font": round(float(b.max_font_size), 2),
                        "bold": bool(b.is_bold),
                        "bbox": [round(float(x), 2) for x in b.bbox],
                        "page": page_number,
                        "page_wh": [round(W, 2), round(H, 2)],
                    }
                )
            return out

        system = "You are a strict PDF block classifier. Output JSON only."

        def make_prompt(items: list[dict]) -> str:
            return f"""
Classify each block from a research paper PDF page.

Return a JSON array with EXACTLY the same number of items as the input.
Each output item MUST be an object with keys:
- i: integer (copy input i)
- action: "keep" or "drop"
- kind: "heading" | "body" | "table" | "math" | "code" | "caption"
- heading_level: 1 | 2 | 3 | null  (only for kind=heading)
- text: string (cleaned text; keep meaning; fix mojibake/ligatures; keep spacing for tables)

STRICT RULES:
1) Headings: kind="heading" ONLY if text matches a real paper section heading:
   - Numbered: ^\\d+(\\.\\d+)*\\s+<LETTER>  (examples: "1 INTRODUCTION", "5.2 Adaptive Control").
   - Appendix: ^[A-Z](?:\\.\\d+)*\\s+<LETTER> (examples: "A DETAILS...", "B.1 ...").
   - The literal word "APPENDIX".
2) Drop boilerplate/noise: headers, footers, page numbers, copyright.
3) Table vs code vs math:
   - table: rows/columns of numbers.
   - math: equations/symbols.
   - code: pseudocode/algorithms (while/for/if, arrows etc).
4) Captions: kind="caption" if starts with "Fig." or "Table".
5) Never invent content.

INPUT JSON:
{json.dumps(items, ensure_ascii=False)}
""".strip()

        batch_size = max(10, int(self.cfg.classify_batch_size))
        all_results: list[dict] = []
        for start in range(0, len(blocks), batch_size):
            sub = blocks[start : start + batch_size]
            items = pack(sub, offset=start)
            if llm_cfg.request_sleep_s > 0:
                time.sleep(llm_cfg.request_sleep_s)
            try:
                resp = self._llm_create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": make_prompt(items)},
                    ],
                    temperature=0.0,
                    max_tokens=llm_cfg.max_tokens,
                )
            except Exception:
                return None
            content = resp.choices[0].message.content or ""
            arr = self._extract_json_array(content)
            if not isinstance(arr, list) or len(arr) != len(items):
                return None
            all_results.extend(arr)
        return all_results

    def call_llm_translate_zh(self, md: str) -> str:
        if not self.cfg.llm or not self._client:
            return md
        llm_cfg = self.cfg.llm
        prompt = (
            "Translate to Chinese. Keep ALL Markdown structure (#, $$, images, code fences) exactly. "
            "Do not translate author names, venues, citations, or LaTeX.\n\n"
            + md
        )
        if llm_cfg.request_sleep_s > 0:
            time.sleep(llm_cfg.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[{"role": "system", "content": "Translator mode."}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=llm_cfg.max_tokens,
            )
        except Exception:
            return md
        return (resp.choices[0].message.content or md).strip()

    def call_llm_split_references(self, ref_block: str, *, paper_name: str) -> Optional[str]:
        if not self.cfg.llm:
            return None
        prompt = (
            f"Split this aggregated references block from paper '{paper_name}' into individual reference items.\n"
            "Return them as a Markdown numbered list.\n"
            "Do not change the text content much, just separate them.\n\n"
            f"BLOCK:\n{ref_block}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "Reference splitter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
        return (resp.choices[0].message.content or "").strip()

    def run_llm_jobs_parallel(self, jobs: list[tuple[str, int, Callable[[], Optional[str]]]]) -> list[Optional[str]]:
        if not jobs:
            return []
        worker_cap = max(1, int(self.cfg.llm_workers))
        if worker_cap <= 1 or len(jobs) <= 1:
            out_seq: list[Optional[str]] = []
            for _, _, fn in jobs:
                try:
                    out_seq.append(fn())
                except Exception:
                    out_seq.append(None)
            return out_seq

        max_workers = min(worker_cap, len(jobs))
        out: list[Optional[str]] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_to_idx = {executor.submit(fn): i for i, (_, _, fn) in enumerate(jobs)}
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                try:
                    out[i] = fut.result()
                except Exception:
                    out[i] = None
        return out
