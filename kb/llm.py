from __future__ import annotations

import time
from typing import Iterator, Optional

from openai import OpenAI

from .config import Settings


class DeepSeekChat:
    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise RuntimeError(
                "缺少 QWEN_API_KEY / DEEPSEEK_API_KEY（或 OPENAI_API_KEY）。请先在环境变量里设置，再启动 UI/脚本。"
            )
        self._settings = settings
        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
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
