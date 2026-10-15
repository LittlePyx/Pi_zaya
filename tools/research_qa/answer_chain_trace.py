"""Opt-in raw answer diagnostics for the isolated acceptance backend only."""
from __future__ import annotations

from functools import wraps
import json
import os
from pathlib import Path
import threading
import time


def install(path: Path) -> None:
    if os.environ.get("KB_TEST_ISOLATED") != "1":
        raise RuntimeError("Raw answer tracing requires an isolated acceptance backend")
    import kb.generation_answer_finalize_runtime as finalize
    import kb.task_runtime as runtime

    lock = threading.Lock()
    path.parent.mkdir(parents=True, exist_ok=True)

    def wrap(module, name: str) -> None:
        original = getattr(module, name)

        @wraps(original)
        def traced(*args, **kwargs):
            inputs = {key: value for key, value in kwargs.items() if key != "settings_obj"}
            event = {
                "stage": name, "thread": threading.get_ident(), "time": time.time(),
                "args": args, "kwargs": inputs,
            }
            # Serialize before execution: finalization may mutate evidence/contract inputs.
            event = json.loads(json.dumps(event, ensure_ascii=False, default=str))
            try:
                result = original(*args, **kwargs)
                event["result"] = result
                return result
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                with lock, path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

        setattr(module, name, traced)

    for module, name in (
        (runtime, "_finalize_generation_answer"),
        (runtime, "_apply_research_facet_quality"),
        (finalize, "audit_and_repair_claim_evidence"),
        (finalize, "_normalize_citation_plan_supported_terms"),
        (finalize, "_build_answer_quality_probe"),
        (runtime, "_build_exact_relation_grounded_fallback"),
        (runtime, "_build_scoped_not_found_answer"),
    ):
        wrap(module, name)
