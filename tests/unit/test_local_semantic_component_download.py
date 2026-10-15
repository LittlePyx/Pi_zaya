from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kb.local_semantic_component import (
    CANDIDATE_ID,
    MODEL_REVISION,
    LocalSemanticComponentError,
    LocalSemanticComponentManifest,
)
from kb.local_semantic_component_download import (
    LocalSemanticComponentDownloader,
    RequestsComponentArchiveTransport,
)


class _Transfer:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        headers: dict[str, str] | None = None,
        final_url: str = "https://github.com/release/component-v1/component.zip",
    ) -> None:
        self.payload = payload
        self.status_code = status
        self.headers = headers or {}
        self.final_url = final_url
        self.closed = False

    def iter_bytes(self):
        for offset in range(0, len(self.payload), 2):
            yield self.payload[offset : offset + 2]

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class _Transport:
    def __init__(self, payload: bytes, *, ignore_range: bool = False) -> None:
        self.payload = payload
        self.ignore_range = ignore_range
        self.calls: list[int] = []

    def open(self, _manifest, *, start: int):
        self.calls.append(start)
        if start and not self.ignore_range:
            return _Transfer(
                self.payload[start:],
                status=206,
                headers={
                    "Content-Range": (
                        f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
                    )
                },
            )
        return _Transfer(self.payload, status=200)


def _manifest(payload: bytes) -> LocalSemanticComponentManifest:
    return LocalSemanticComponentManifest.from_payload(
        {
            "component_id": "component",
            "release_id": "component-v1",
            "download_url": (
                "https://github.com/release/component-v1/component.zip"
            ),
            "archive_name": "component.zip",
            "archive_root": "c",
            "archive_bytes": len(payload),
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_bytes": 1,
            "payload_file_count": 1,
            "payload_sha256": "0" * 64,
            "candidate_id": CANDIDATE_ID,
            "model_revision": MODEL_REVISION,
        },
        allowed_hosts={"github.com"},
    )


def test_download_is_verified_atomic_and_reused(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    progress = []
    downloader = LocalSemanticComponentDownloader(tmp_path)
    first = downloader.download(
        _manifest(payload),
        _Transport(payload),
        on_progress=progress.append,
    )
    second_transport = _Transport(payload)
    second = downloader.download(_manifest(payload), second_transport)

    assert first.archive_path.read_bytes() == payload
    assert first.downloaded_bytes == len(payload) and first.reused is False
    assert second.reused is True and second_transport.calls == []
    assert progress[-1].completed_bytes == len(payload)
    assert not list(tmp_path.glob("*.part"))


def test_cancel_retains_partial_and_valid_range_resumes(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    downloader = LocalSemanticComponentDownloader(tmp_path)
    cancel = {"value": False}

    def on_progress(item) -> None:
        if item.completed_bytes >= 2:
            cancel["value"] = True

    with pytest.raises(LocalSemanticComponentError) as captured:
        downloader.download(
            _manifest(payload),
            _Transport(payload),
            should_cancel=lambda: cancel["value"],
            on_progress=on_progress,
        )
    assert captured.value.code == "semantic_component_cancelled"

    partial = next(tmp_path.glob("*.part"))
    assert partial.read_bytes() == b"ab"
    transport = _Transport(payload)
    result = downloader.download(_manifest(payload), transport)
    assert transport.calls == [2]
    assert result.resumed_bytes == 2
    assert result.archive_path.read_bytes() == payload


def test_invalid_range_response_discards_partial_and_restarts(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    manifest = _manifest(payload)
    partial = tmp_path / f"{manifest.archive_sha256}.zip.part"
    partial.write_bytes(b"ab")
    transport = _Transport(payload, ignore_range=True)

    result = LocalSemanticComponentDownloader(tmp_path).download(manifest, transport)

    assert transport.calls == [2, 0]
    assert result.resumed_bytes == 0
    assert result.archive_path.read_bytes() == payload


def test_hash_mismatch_deletes_untrusted_partial(tmp_path: Path) -> None:
    expected = b"expected"
    downloader = LocalSemanticComponentDownloader(tmp_path)
    with pytest.raises(LocalSemanticComponentError) as captured:
        downloader.download(_manifest(expected), _Transport(b"bad-data"))
    assert captured.value.code == "component_archive_hash_mismatch"
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob("*.zip"))


def test_oversized_response_is_stopped(tmp_path: Path) -> None:
    expected = b"ok"
    with pytest.raises(LocalSemanticComponentError) as captured:
        LocalSemanticComponentDownloader(tmp_path).download(
            _manifest(expected),
            _Transport(b"too-large"),
        )
    assert captured.value.code == "component_download_response_too_large"
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob("*.zip"))


def test_requests_transport_rejects_redirect_to_unallowlisted_host() -> None:
    class Response:
        status_code = 200
        headers = {}
        url = "https://evil.example/component.zip"
        closed = False

        def close(self) -> None:
            self.closed = True

    response = Response()
    transport = RequestsComponentArchiveTransport(
        allowed_hosts={"github.com"},
        request_get=lambda *_args, **_kwargs: response,
    )
    with pytest.raises(LocalSemanticComponentError) as captured:
        transport.open(_manifest(b"ok"), start=0)
    assert captured.value.code == "component_download_redirect_not_allowed"
    assert response.closed is True
