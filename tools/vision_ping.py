from __future__ import annotations

import os
import sys
import base64
import struct
import zlib


def main() -> int:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        print(f"[vision_ping] openai import failed: {e}", file=sys.stderr)
        return 2

    api_key = (os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("[vision_ping] Missing QWEN_API_KEY (or OPENAI_API_KEY).", file=sys.stderr)
        return 2

    base_url = (os.environ.get("QWEN_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip().rstrip("/")
    model = (os.environ.get("QWEN_MODEL") or os.environ.get("OPENAI_MODEL") or "qwen3-vl-plus").strip()

    def _crc32(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + _crc32(tag, data)

    def _make_png(w: int, h: int) -> bytes:
        # RGBA, solid white. No external deps.
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
        # Each scanline starts with filter type 0x00.
        row = b"\x00" + (b"\xFF\xFF\xFF\xFF" * w)
        raw = row * h
        comp = zlib.compress(raw, level=6)
        return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", comp) + _chunk(b"IEND", b"")

    png = _make_png(32, 32)  # Qwen VL requires both width/height > 10 px.
    data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with exactly: OK"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=50,
            timeout=60,
        )
    except Exception as e:
        print(f"[vision_ping] FAILED model={model!r} base_url={base_url!r} err={e}", file=sys.stderr)
        return 1

    out = (resp.choices[0].message.content or "").strip()
    print(f"[vision_ping] OK model={model!r} base_url={base_url!r} out={out!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

