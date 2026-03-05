from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any, Iterator, Optional

from openai import OpenAI

from .config import Settings


def _has_multimodal_content(messages: list[dict]) -> bool:
    for msg in list(messages or []):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip().lower()
            if part_type in {"image_url", "input_image", "image"}:
                return True
    return False


class DeepSeekChat:
    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise RuntimeError(
                "缺少 QWEN_API_KEY / DEEPSEEK_API_KEY（或 OPENAI_API_KEY）。请先在环境变量里设置，再启动 UI/脚本。"
            )
        self._settings = settings
        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def _resolve_hard_timeout_s(self, *, request_timeout_s: float) -> float:
        try:
            raw = str(os.environ.get("KB_LLM_HARD_TIMEOUT_S", "") or "").strip()
            if raw:
                val = float(raw)
                return max(0.0, min(600.0, val))
        except Exception:
            pass
        try:
            base = float(request_timeout_s or 0.0)
        except Exception:
            base = 0.0
        if base <= 0:
            return 0.0
        return max(18.0, min(120.0, max(base + 6.0, base * 1.2)))

    def _create_with_guard_timeout(self, **kwargs):
        timeout_s = float(kwargs.get("timeout", 0.0) or 0.0)
        hard_timeout_s = self._resolve_hard_timeout_s(request_timeout_s=timeout_s)
        if hard_timeout_s <= 0:
            return self._client.chat.completions.create(**kwargs)

        q: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)

        def _run_create() -> None:
            try:
                resp = self._client.chat.completions.create(**kwargs)
                try:
                    q.put_nowait(("ok", resp))
                except Exception:
                    pass
            except Exception as e:
                try:
                    q.put_nowait(("err", e))
                except Exception:
                    pass

        t = threading.Thread(
            target=_run_create,
            name="kb_llm_chat_guard",
            daemon=True,
        )
        t.start()
        try:
            kind, payload = q.get(timeout=max(1.0, float(hard_timeout_s)))
        except queue.Empty:
            raise TimeoutError(
                f"LLM hard timeout after {float(hard_timeout_s):.1f}s "
                f"(sdk_timeout={float(timeout_s):.1f}s)"
            )

        if kind == "err":
            if isinstance(payload, Exception):
                raise payload
            raise RuntimeError(str(payload))
        return payload

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                resp = self._create_with_guard_timeout(
                    model=self._settings.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self._settings.timeout_s,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= self._settings.max_retries:
                    break
                time.sleep(0.6 * (attempt + 1))
        raise last_err  # type: ignore[misc]

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Iterator[str]:
        """
        Stream assistant output incrementally.

        Notes:
        - We only stream the final text (no chain-of-thought).
        - If streaming fails upstream, callers should fallback to .chat().
        """
        if _has_multimodal_content(messages):
            # Some OpenAI-compatible providers accept multimodal inputs in non-stream mode
            # but silently degrade/ignore images when stream=True.
            text = self.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
            if text:
                yield text
            return

        resp = self._client.chat.completions.create(
            model=self._settings.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self._settings.timeout_s,
            stream=True,
        )

        for event in resp:
            try:
                choice0 = event.choices[0]
                delta = getattr(choice0, "delta", None)
                piece = ""
                if delta is not None:
                    piece = (getattr(delta, "content", None) or "")
                if piece:
                    yield piece
            except Exception:
                continue
