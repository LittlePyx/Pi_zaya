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


def _plain_annotation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value
    elif hasattr(value, "model_dump"):
        try:
            raw = value.model_dump()
        except Exception:
            raw = {}
    else:
        raw = {}
        for key in ("type", "url", "title", "start_index", "end_index", "url_citation"):
            item = getattr(value, key, None)
            if item is not None:
                raw[key] = item
    if not isinstance(raw, dict) or not raw:
        return None
    citation = raw.get("url_citation") if isinstance(raw.get("url_citation"), dict) else raw
    return {
        "type": str(raw.get("type") or "").strip(),
        "url": str(citation.get("url") or "").strip(),
        "title": str(citation.get("title") or "").strip(),
        "start_index": citation.get("start_index", raw.get("start_index")),
        "end_index": citation.get("end_index", raw.get("end_index")),
    }


def _plain_annotations(message: Any) -> list[dict[str, Any]]:
    annotations = getattr(message, "annotations", None)
    if annotations is None and isinstance(message, dict):
        annotations = message.get("annotations")
    out: list[dict[str, Any]] = []
    for item in list(annotations or []):
        row = _plain_annotation(item)
        if row and (row.get("url") or row.get("title")):
            out.append(row)
    return out[:12]


class DeepSeekChat:
    def __init__(self, settings: Settings) -> None:
        if not settings.text_api_key and not getattr(settings, "agent_web_search_api_key", None):
            raise RuntimeError(
                "缺少 QWEN_API_KEY / DEEPSEEK_API_KEY（或 OPENAI_API_KEY）。请先在环境变量里设置，再启动 UI/脚本。"
            )
        self._settings = settings
        self._text_client = (
            OpenAI(api_key=settings.text_api_key, base_url=settings.text_base_url, max_retries=0)
            if settings.text_api_key
            else None
        )
        if settings.auto_route:
            self._vision_client = OpenAI(
                api_key=settings.vision_api_key,
                base_url=settings.vision_base_url,
                max_retries=0,
            )
        else:
            self._vision_client = self._text_client

    # ------------------------------------------------------------------
    # Backward-compatible accessors (existing code may read these).
    # ------------------------------------------------------------------
    @property
    def _client(self) -> OpenAI:
        if self._text_client is None:
            raise RuntimeError("Text model API key is not configured.")
        return self._text_client

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    def _select_model(self, messages: list[dict]) -> tuple[OpenAI, str]:
        if self._settings.auto_route and _has_multimodal_content(messages):
            return self._vision_client, self._settings.vision_model
        if self._text_client is None:
            raise RuntimeError("Text model API key is not configured.")
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
        *,
        timeout_s: float | None = None,
        max_retries: int | None = None,
    ) -> str:
        client, model = self._select_model(messages)
        request_timeout_s = float(timeout_s) if timeout_s is not None else float(self._settings.timeout_s)
        retry_count = int(max_retries) if max_retries is not None else int(self._settings.max_retries)
        retry_count = max(0, retry_count)
        last_err: Optional[Exception] = None
        for attempt in range(retry_count + 1):
            try:
                resp = self._create_with_guard_timeout(
                    client=client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout_s,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= retry_count:
                    break
                time.sleep(0.6 * (attempt + 1))
        raise last_err  # type: ignore[misc]

    def chat_with_web_search(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        if not getattr(self._settings, "agent_web_search_enabled", False):
            raise RuntimeError("Agent web search is disabled.")
        api_key = getattr(self._settings, "agent_web_search_api_key", None)
        if not api_key:
            raise RuntimeError("Agent web search API key is not configured.")
        model = str(getattr(self._settings, "agent_web_search_model", "") or "").strip()
        if not model:
            raise RuntimeError("Agent web search model is not configured.")
        base_url = str(
            getattr(self._settings, "agent_web_search_base_url", "")
            or "https://api.openai.com/v1"
        ).strip().rstrip("/")
        context_size = str(getattr(self._settings, "agent_web_search_context_size", "") or "low").strip().lower()
        if context_size not in {"low", "medium", "high"}:
            context_size = "low"
        client = OpenAI(api_key=api_key, base_url=base_url)
        last_err: Optional[Exception] = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                resp = self._create_with_guard_timeout(
                    client=client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    web_search_options={"search_context_size": context_size},
                    timeout=self._settings.timeout_s,
                )
                message = resp.choices[0].message
                return {
                    "content": (getattr(message, "content", None) or "").strip(),
                    "annotations": _plain_annotations(message),
                    "model": model,
                }
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
        if _has_multimodal_content(messages):
            text = self.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
            if text:
                yield text
            return

        client, model = self._select_model(messages)
        routes: list[tuple[OpenAI, str, float, str]] = [
            (client, model, self._primary_stream_timeout_s(), "text")
        ]
        if (
            bool(getattr(self._settings, "auto_route", False))
            and self._vision_client is not None
            and self._vision_client is not client
        ):
            routes.append(
                (
                    self._vision_client,
                    str(getattr(self._settings, "vision_model", "") or "").strip(),
                    float(self._settings.timeout_s),
                    "vision",
                )
            )

        last_err: Optional[Exception] = None
        for route_index, (route_client, route_model, route_timeout, route_kind) in enumerate(routes):
            emitted = False
            try:
                for piece in self._stream_route_with_visible_deadlines(
                    client=route_client,
                    route_kind=route_kind,
                    model=route_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout_s=route_timeout,
                ):
                    emitted = True
                    yield piece
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if emitted or route_index >= len(routes) - 1:
                    raise
        if last_err is not None:
            raise last_err

    def _stream_route_with_visible_deadlines(
        self,
        *,
        client: OpenAI,
        route_kind: str,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        request_timeout_s: float,
    ) -> Iterator[str]:
        """Stream one route with wall-clock deadlines for visible output.

        Provider streams may send HTTP keepalives or empty deltas indefinitely,
        so the SDK read timeout alone cannot bound the wait users see.  The
        request runs on a daemon worker and only non-empty text resets the
        visible-output deadline.
        """

        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        stop = threading.Event()
        resources: dict[str, Any] = {}

        def run() -> None:
            route_client = client
            owns_client = False
            try:
                # Real SDK clients are recreated inside the worker.  Sharing a
                # client created on another thread previously caused httpx pool
                # corruption after a timed-out request.  Lightweight fake
                # clients used by tests are intentionally reused.
                if isinstance(client, OpenAI):
                    if route_kind == "vision":
                        api_key = self._settings.vision_api_key
                        base_url = self._settings.vision_base_url
                    else:
                        api_key = self._settings.text_api_key
                        base_url = self._settings.text_base_url
                    route_client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
                    owns_client = True
                resources["client"] = route_client
                resp = route_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout_s,
                    stream=True,
                )
                resources["response"] = resp
                for event in resp:
                    if stop.is_set():
                        break
                    try:
                        choice0 = event.choices[0]
                        delta = getattr(choice0, "delta", None)
                        piece = (getattr(delta, "content", None) or "") if delta is not None else ""
                    except Exception:
                        continue
                    if piece:
                        events.put(("piece", piece))
                events.put(("done", None))
            except Exception as exc:  # noqa: BLE001
                events.put(("error", exc))
            finally:
                if owns_client:
                    try:
                        route_client.close()
                    except Exception:
                        pass

        worker = threading.Thread(target=run, name=f"llm-stream-{route_kind}", daemon=True)
        worker.start()
        emitted = False
        try:
            while True:
                visible_timeout_s = (
                    self._stream_idle_timeout_s()
                    if emitted
                    else self._first_visible_token_timeout_s()
                )
                try:
                    kind, payload = events.get(timeout=visible_timeout_s)
                except queue.Empty as exc:
                    phase = "idle visible output" if emitted else "first visible token"
                    raise TimeoutError(
                        f"LLM stream timed out waiting for {phase} after {visible_timeout_s:.1f}s"
                    ) from exc
                if kind == "piece":
                    emitted = True
                    yield str(payload)
                elif kind == "done":
                    return
                elif kind == "error":
                    raise payload
        finally:
            stop.set()
            for name in ("response", "client"):
                resource = resources.get(name)
                close = getattr(resource, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

    def _primary_stream_timeout_s(self) -> float:
        configured = str(os.environ.get("KB_LLM_PRIMARY_STREAM_TIMEOUT_S", "25") or "25").strip()
        try:
            timeout_s = float(configured)
        except Exception:
            timeout_s = 25.0
        return max(8.0, min(float(self._settings.timeout_s), timeout_s))

    def _first_visible_token_timeout_s(self) -> float:
        configured = str(os.environ.get("KB_LLM_FIRST_TOKEN_TIMEOUT_S", "12") or "12").strip()
        try:
            timeout_s = float(configured)
        except Exception:
            timeout_s = 12.0
        return max(3.0, min(float(self._settings.timeout_s), timeout_s))

    def _stream_idle_timeout_s(self) -> float:
        configured = str(os.environ.get("KB_LLM_STREAM_IDLE_TIMEOUT_S", "25") or "25").strip()
        try:
            timeout_s = float(configured)
        except Exception:
            timeout_s = 25.0
        return max(8.0, min(float(self._settings.timeout_s), timeout_s))
