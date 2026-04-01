from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import List, Optional

try:
    import fitz
except ImportError:
    fitz = None

from .page_vision_direct_page import process_vision_direct_page


def process_batch_vision_direct(self, doc, pdf_path: Path, assets_dir: Path, speed_mode: str = "normal") -> List[Optional[str]]:
    """
    Bypass all text-extraction / block-classification logic.
    For every page: render a high-DPI screenshot, send it to the vision LLM,
    and collect the Markdown it returns.
    Supports parallel processing with ThreadPoolExecutor.
    """
    total_pages = len(doc)
    results: List[Optional[str]] = [None] * total_pages

    start = max(0, int(getattr(self.cfg, "start_page", 0) or 0))
    end = int(getattr(self.cfg, "end_page", -1) or -1)
    if end < 0:
        end = total_pages
    end = min(total_pages, end)
    if start >= end:
        return results

    import multiprocessing

    speed_config = self._get_speed_mode_config(speed_mode, total_pages)
    cpu_count = multiprocessing.cpu_count()

    base_dpi = int(getattr(self, "dpi", 200) or 200)
    try:
        vision_dpi = int(os.environ.get("KB_PDF_VISION_DPI", "") or "")
        if vision_dpi > 0:
            dpi = max(200, min(600, vision_dpi))
        else:
            dpi = speed_config.get("dpi", 220)
    except Exception:
        dpi = speed_config.get("dpi", 220)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    max_parallel = speed_config.get("max_parallel_pages", min(8, max(1, cpu_count)))
    try:
        max_parallel = int(max_parallel)
    except Exception:
        max_parallel = min(8, max(1, cpu_count))
    max_parallel = max(1, min(64, max_parallel, total_pages))

    raw_llm_pw = (os.environ.get("KB_PDF_LLM_PAGE_WORKERS") or "").strip()
    num_workers = int(raw_llm_pw) if raw_llm_pw else int(os.environ.get("KB_PDF_WORKERS", "0") or "0")
    if num_workers <= 0:
        if total_pages <= 2:
            num_workers = 1
        else:
            num_workers = min(max_parallel, cpu_count, total_pages)

    inflight_source = "speed_config"
    try:
        raw_inflight = (os.environ.get("KB_LLM_MAX_INFLIGHT") or "").strip()
        if raw_inflight:
            max_inflight = int(raw_inflight)
            inflight_source = "env:KB_LLM_MAX_INFLIGHT"
        else:
            worker_inflight = 0
            try:
                worker_inflight = int(self.llm_worker.get_llm_max_inflight())
            except Exception:
                try:
                    worker_inflight = int(getattr(self.llm_worker, "_llm_max_inflight", 0) or 0)
                except Exception:
                    worker_inflight = 0
            if worker_inflight > 0:
                max_inflight = worker_inflight
                inflight_source = "llm_worker"
            else:
                max_inflight = int(speed_config.get("max_inflight", 8) or 8)
        max_inflight = max(1, min(32, int(max_inflight)))
    except Exception:
        max_inflight = max(1, min(32, int(speed_config.get("max_inflight", 8) or 8)))
        inflight_source = "fallback"

    num_workers_before_cap = num_workers
    cap = None

    if raw_llm_pw:
        cap = None
        num_workers = min(int(num_workers), int(total_pages))
    else:
        try:
            raw_cap = (os.environ.get("KB_PDF_LLM_PAGE_WORKERS_CAP") or "").strip()
            if raw_cap:
                cap = int(raw_cap)
                cap = max(1, min(64, int(cap)))
            else:
                cap = max(1, min(int(max_parallel), int(max_inflight)))

            if cap is not None:
                num_workers = min(int(num_workers), int(cap), int(total_pages))
            else:
                num_workers = min(int(num_workers), int(total_pages))
        except Exception:
            cap = max(1, min(int(max_parallel), int(max_inflight)))
            num_workers = min(int(num_workers), int(cap), int(total_pages))

    try:
        print(
            f"[VISION_DIRECT] worker calculation: raw_llm_pw={raw_llm_pw!r}, "
            f"num_workers_before={num_workers_before_cap}, final_num_workers={num_workers}, "
            f"max_inflight={max_inflight} ({inflight_source}), cap={cap}, total_pages={total_pages}",
            flush=True,
        )
        if max_inflight < num_workers:
            print(
                f"[VISION_DIRECT] WARNING: KB_LLM_MAX_INFLIGHT={max_inflight} < num_workers={num_workers}. "
                f"This may cause timeout errors. Consider setting KB_LLM_MAX_INFLIGHT >= {num_workers}",
                flush=True,
            )
    except Exception:
        pass

    if num_workers <= 1 or total_pages <= 1:
        print(f"[VISION_DIRECT] Converting pages {start+1}-{end} via VL screenshots (dpi={dpi}, sequential)", flush=True)
        for i in range(start, end):
            t0 = time.time()
            print(f"Processing page {i+1}/{total_pages} (vision-direct) ...", flush=True)
            try:
                page = doc.load_page(i)
                results[i] = process_vision_direct_page(
                    self,
                    page=page,
                    page_index=i,
                    total_pages=total_pages,
                    pdf_path=pdf_path,
                    assets_dir=assets_dir,
                    speed_mode=speed_mode,
                    speed_config=speed_config,
                    dpi=dpi,
                    mat=mat,
                    started_at=t0,
                )
            except Exception as e:
                error_str = str(e)
                print(f"[VISION_DIRECT] error page {i+1}: {e}", flush=True)
                if "Access denied" in error_str or "account is in good standing" in error_str:
                    print("[VISION_DIRECT] API access denied detected; check account status.", flush=True)
                import traceback

                traceback.print_exc()
                results[i] = None
        return results

    print(f"[VISION_DIRECT] Converting pages {start+1}-{end} via VL screenshots (dpi={dpi}, {num_workers} workers)", flush=True)
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    worker_docs = threading.local()
    opened_docs = []
    opened_docs_lock = threading.Lock()

    def _get_worker_doc():
        local_doc = getattr(worker_docs, "doc", None)
        if local_doc is None:
            local_doc = fitz.open(str(pdf_path))
            worker_docs.doc = local_doc
            with opened_docs_lock:
                opened_docs.append(local_doc)
        return local_doc

    def process_single_page(i: int):
        try:
            print(f"Processing page {i+1}/{total_pages} (vision-direct) ...", flush=True)
            t0 = time.time()
            local_doc = _get_worker_doc()
            page = local_doc.load_page(i)
            result = process_vision_direct_page(
                self,
                page=page,
                page_index=i,
                total_pages=total_pages,
                pdf_path=pdf_path,
                assets_dir=assets_dir,
                speed_mode=speed_mode,
                speed_config=speed_config,
                dpi=dpi,
                mat=mat,
                started_at=t0,
            )
            return i, result
        except Exception as e:
            error_str = str(e)
            print(f"[VISION_DIRECT] error page {i+1}: {e}", flush=True)
            if "Access denied" in error_str or "account is in good standing" in error_str:
                print("[VISION_DIRECT] API access denied detected; check account status.", flush=True)
            import traceback

            traceback.print_exc()
            return i, None

    executor = ThreadPoolExecutor(max_workers=num_workers)
    futures = {executor.submit(process_single_page, i): i for i in range(start, end)}
    pending = set(futures.keys())

    hb_every_s = 8.0
    try:
        hb_every_s = float(os.environ.get("KB_PDF_BATCH_HEARTBEAT_S", str(hb_every_s)) or hb_every_s)
        hb_every_s = max(2.0, min(60.0, hb_every_s))
    except Exception:
        hb_every_s = 8.0
    last_hb = time.time()

    try:
        while pending:
            now_ts = time.time()
            done, not_done = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            pending = set(not_done)
            try:
                if (now_ts - last_hb) >= hb_every_s:
                    inflight_pages = sorted({int(futures[fut]) + 1 for fut in pending})
                    if inflight_pages:
                        head = inflight_pages[:12]
                        more = len(inflight_pages) - len(head)
                        extra = f" (+{more} more)" if more > 0 else ""
                        print(
                            f"[VISION_DIRECT] still running pages: {head}{extra} | workers={num_workers} llm_inflight={max_inflight}",
                            flush=True,
                        )
                    last_hb = now_ts
            except Exception:
                pass
            for future in done:
                i = futures[future]
                try:
                    i2, result = future.result()
                    results[i2] = result
                except Exception as e:
                    print(f"[VISION_DIRECT] error processing page {i+1}: {e}", flush=True)
                    results[i] = None
    finally:
        if pending:
            for future in pending:
                i = futures.get(future)
                if i is not None and results[i] is None:
                    results[i] = f"<!-- kb_page: {i+1} -->\n\n[Page {i+1} conversion incomplete]"
                try:
                    future.cancel()
                except Exception:
                    pass
        executor.shutdown(wait=True, cancel_futures=True)
        for local_doc in opened_docs:
            try:
                local_doc.close()
            except Exception:
                pass

    return results
