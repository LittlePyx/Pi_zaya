from __future__ import annotations

import json
import re
import time
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

class LLMWorker:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg
        self._client = None
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

    def _llm_create(self, **kwargs):
        if not self._client:
            raise RuntimeError("LLM client not initialized")
        # Retry logic could go here
        return self._client.chat.completions.create(model=self.cfg.llm.model, **kwargs)

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
        
        ctx_prompt = ""
        if context_before:
            ctx_prompt += f"\nContext before:\n...{context_before[-300:]}\n"
        if context_after:
            ctx_prompt += f"\nContext after:\n{context_after[:300]}...\n"
            
        eq_hint = f"(Equation number: {eq_number})" if eq_number else ""
        prompt = (
            f"Recover this garbled math equation from PDF page {page_number+1} {eq_hint}.\n"
            "Return ONLY the LaTeX code (without match $ delimiters unless it is inline).\n"
            "If it's a display equation, ensure it is standard LaTeX.\n"
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
        # Strip $$...$$ or \[...\]
        if out.startswith("$$") and out.endswith("$$"):
            out = out[2:-2].strip()
        elif out.startswith("\\[") and out.endswith("\\]"):
            out = out[2:-2].strip()
        return out

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
