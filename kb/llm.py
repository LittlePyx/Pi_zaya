from __future__ import annotations

import os
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
        if not settings.text_api_key:
            raise RuntimeError(
                "缺少 QWEN_API_KEY / DEEPSEEK_API_KEY（或 OPENAI_API_KEY）。请先在环境变量里设置，再启动 UI/脚本。"
            )
        self._settings = settings
        self._text_client = OpenAI(
            api_key=settings.text_api_key, base_url=settings.text_base_url
        )
        if settings.auto_route:
            self._vision_client = OpenAI(
                api_key=settings.vision_api_key, base_url=settings.vision_base_url
            )
        else:
            self._vision_client = self._text_client

    # ------------------------------------------------------------------
    # Backward-compatible accessors (existing code may read these).
    # ------------------------------------------------------------------
    @property
    def _client(self) -> OpenAI:
        return self._text_client

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    def _select_model(self, messages: list[dict]) -> tuple[OpenAI, str]:
        if self._settings.auto_route and _has_multimodal_content(messages):
            return self._vision_client, self._settings.vision_model
        return self._text_client, self._settings.text_model

    # ------------------------------------------------------------------
    # Timeout guard  (deprecated — the SDK timeout is reliable on its own)
    # ------------------------------------------------------------------

    def _create_with_guard_timeout(self, *, client: OpenAI, **kwargs):
        # The native SDK timeout mechanism is reliable.  We previously used a
        # guard-thread wrapper here, but that caused httpx connection-pool
        # corruption when the client was created on the main thread and invoked
        # from the guard thread — the API call would succeed but return empty
        # content.  Delegate to the SDK's own timeout instead.
        return client.chat.completions.create(**kwargs)

    # ------------------------------------------------------------------
    # Chat (non-streaming)
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        client, model = self._select_model(messages)
        last_err: Optional[Exception] = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                resp = self._create_with_guard_timeout(
                    client=client,
                    model=model,
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

    # ------------------------------------------------------------------
    # Chat (streaming)
    # ------------------------------------------------------------------
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
        - Multimodal messages use non-stream mode (some providers silently
          degrade images under stream=True).
        """
        client, model = self._select_model(messages)

        if _has_multimodal_content(messages):
            text = self.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
            if text:
                yield text
            return

        resp = client.chat.completions.create(
            model=model,
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
