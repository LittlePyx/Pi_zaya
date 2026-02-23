from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from starlette.responses import StreamingResponse


async def sse_generator(
    poll_fn,
    *,
    interval: float = 0.15,
    done_key: str = "done",
) -> AsyncGenerator[str, None]:
    while True:
        data = poll_fn()
        yield f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
        if data and data.get(done_key):
            return
        await asyncio.sleep(interval)


def sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
