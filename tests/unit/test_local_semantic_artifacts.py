from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kb.local_semantic_artifacts import (
    ArtifactPreparationCancelled,
    ArtifactPreparationError,
    LocalSemanticArtifact,
    RequestsArtifactTransport,
    VerifiedLocalSemanticArtifacts,
)


class _Transfer:
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]) -> None:
        self.payload = payload
        self.status_code = status
        self.headers = headers
        self.closed = False

    def iter_bytes(self):
        for offset in range(0, len(self.payload), 2):
            yield self.payload[offset : offset + 2]

    def close(self) -> None:
        self.closed = True


class _Transport:
    def __init__(self, payloads: dict[str, bytes], *, ignore_range: bool = False) -> None:
        self.payloads = payloads
        self.ignore_range = ignore_range
        self.calls: list[tuple[str, int]] = []

    def open(self, artifact: LocalSemanticArtifact, *, start: int) -> _Transfer:
        self.calls.append((artifact.local_name, start))
        payload = self.payloads[artifact.local_name]
        if start and not self.ignore_range:
            return _Transfer(
                payload[start:],
                status=206,
                headers={"Content-Range": f"bytes {start}-{len(payload) - 1}/{len(payload)}"},
            )
        return _Transfer(payload, status=200, headers={})


def _artifact(name: str, payload: bytes) -> LocalSemanticArtifact:
    return LocalSemanticArtifact(
        repository_path=f"onnx/{name}",
        local_name=name,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_verified_download_is_atomic_reusable_and_reports_progress(tmp_path: Path) -> None:
    payloads = {"a.bin": b"abcdef", "b.bin": b"1234"}
    artifacts = tuple(_artifact(name, payload) for name, payload in payloads.items())
    manager = VerifiedLocalSemanticArtifacts(tmp_path, artifacts=artifacts)
    progress = []
    first = manager.prepare(_Transport(payloads), on_progress=progress.append)
    second_transport = _Transport(payloads)
    second = manager.prepare(second_transport)

    assert manager.is_ready() is True
    assert first.downloaded_bytes == 10 and first.reused_bytes == 0
    assert second.downloaded_bytes == 0 and second.reused_bytes == 10
    assert second_transport.calls == []
    assert progress[-1].completed_units == 2
    assert progress[-1].completed_bytes == 10
    assert not list(tmp_path.rglob("*.part"))


def test_cancel_retains_partial_and_valid_range_resumes_it(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    artifact = _artifact("a.bin", payload)
    manager = VerifiedLocalSemanticArtifacts(tmp_path, artifacts=(artifact,))
    cancel = {"value": False}

    def on_progress(item) -> None:
        if item.completed_bytes >= 2:
            cancel["value"] = True

    with pytest.raises(ArtifactPreparationCancelled):
        manager.prepare(
            _Transport({"a.bin": payload}),
            should_cancel=lambda: cancel["value"],
            on_progress=on_progress,
        )
    partial = tmp_path / "a.bin.part"
    assert partial.read_bytes() == b"ab"

    transport = _Transport({"a.bin": payload})
    result = manager.prepare(transport)
    assert transport.calls == [("a.bin", 2)]
    assert result.resumed_bytes == 2
    assert (tmp_path / "a.bin").read_bytes() == payload
    assert not partial.exists()


def test_invalid_range_response_restarts_from_zero(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    artifact = _artifact("a.bin", payload)
    manager = VerifiedLocalSemanticArtifacts(tmp_path, artifacts=(artifact,))
    (tmp_path / "a.bin.part").write_bytes(b"ab")
    transport = _Transport({"a.bin": payload}, ignore_range=True)

    manager.prepare(transport)

    assert transport.calls == [("a.bin", 2), ("a.bin", 0)]
    assert (tmp_path / "a.bin").read_bytes() == payload


def test_hash_mismatch_never_replaces_existing_file(tmp_path: Path) -> None:
    expected = b"expected"
    artifact = _artifact("a.bin", expected)
    destination = tmp_path / "a.bin"
    destination.write_bytes(b"existing-invalid")
    manager = VerifiedLocalSemanticArtifacts(tmp_path, artifacts=(artifact,))

    with pytest.raises(ArtifactPreparationError) as captured:
        manager.prepare(_Transport({"a.bin": b"bad-data"}))

    assert captured.value.code == "artifact_integrity_mismatch"
    assert destination.read_bytes() == b"existing-invalid"
    assert not manager.ready_path.exists()


def test_oversized_response_is_stopped_before_publication(tmp_path: Path) -> None:
    artifact = _artifact("a.bin", b"ok")
    manager = VerifiedLocalSemanticArtifacts(tmp_path, artifacts=(artifact,))
    with pytest.raises(ArtifactPreparationError) as captured:
        manager.prepare(_Transport({"a.bin": b"too-large"}))
    assert captured.value.code == "artifact_response_too_large"
    assert not (tmp_path / "a.bin").exists()


def test_requests_transport_sends_bounded_immutable_range_request() -> None:
    calls = []

    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 2-3/4"}

        def iter_content(self, *, chunk_size: int):
            assert chunk_size == 1024 * 1024
            yield b"34"

        def close(self) -> None:
            return None

    def get(url: str, **kwargs):
        calls.append((url, kwargs))
        return Response()

    transport = RequestsArtifactTransport(request_get=get)
    artifact = _artifact("a.bin", b"1234")
    transfer = transport.open(artifact, start=2)

    assert list(transfer.iter_bytes()) == [b"34"]
    assert f"/resolve/{artifact.url.split('/resolve/')[1].split('/')[0]}/" in calls[0][0]
    assert calls[0][1] == {
        "headers": {"Accept-Encoding": "identity", "Range": "bytes=2-"},
        "stream": True,
        "timeout": (15.0, 120.0),
        "allow_redirects": True,
    }


def test_transport_failure_is_categorized_without_changing_partial(tmp_path: Path) -> None:
    payload = b"abcd"
    artifact = _artifact("a.bin", payload)
    partial = tmp_path / "a.bin.part"
    partial.write_bytes(b"ab")

    class FailedTransport:
        def open(self, _artifact, *, start: int):
            assert start == 2
            raise TimeoutError("private network detail")

    with pytest.raises(ArtifactPreparationError) as captured:
        VerifiedLocalSemanticArtifacts(tmp_path, artifacts=(artifact,)).prepare(
            FailedTransport()
        )
    assert captured.value.code == "artifact_transport_failed"
    assert partial.read_bytes() == b"ab"
