import base64
from pathlib import Path

import pytest

from kb.path_safety import (
    clean_file_source_path_input,
    image_mime_for_ext,
    is_probably_pdf_bytes,
    path_is_within_roots,
    resolve_existing_file_under_roots,
    resolve_verified_chat_image_upload_path,
    resolve_verified_pdf_file_under_roots,
    sniff_image_ext,
    sniff_image_mime,
    unique_resolved_roots,
    verified_image_bytes_mime,
    verified_image_file_mime,
)

TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)


def test_clean_file_source_path_input_accepts_browser_file_url_variants() -> None:
    raw = "file:///C:/Papers/A%23B%20Paper.en.md?download=1#reader"

    assert clean_file_source_path_input(raw) == "C:/Papers/A#B Paper.en.md"
    assert clean_file_source_path_input("C:/Papers/A#B Paper.en.md") == "C:/Papers/A#B Paper.en.md"
    assert clean_file_source_path_input("C:/Papers/Paper.en.md#reader") == "C:/Papers/Paper.en.md"


def test_sniff_image_mime_uses_content_signature_not_suffix() -> None:
    data = b"\x89PNG\r\n\x1a\nnot-a-real-full-image"

    assert sniff_image_ext(data) == ".png"
    assert sniff_image_mime(data) == "image/png"
    assert image_mime_for_ext(".jpg") == "image/jpeg"


def test_verified_image_mime_requires_decodable_image_content() -> None:
    assert verified_image_bytes_mime(TINY_PNG_BYTES) == "image/png"
    assert verified_image_bytes_mime(b"\x89PNG\r\n\x1a\nnot-a-real-full-image") == ""


def test_resolve_verified_chat_image_upload_path_rejects_fake_image_under_root(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    image_dir = db_dir / "_chat_uploads" / "images"
    image_dir.mkdir(parents=True)
    fake = image_dir / "fake.png"
    fake.write_bytes(b"not really an image")
    fake_header = image_dir / "fake-header.png"
    fake_header.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-full-image")
    valid = image_dir / "valid.png"
    valid.write_bytes(TINY_PNG_BYTES)

    assert resolve_verified_chat_image_upload_path(fake, db_dir=db_dir) is None
    assert resolve_verified_chat_image_upload_path(fake_header, db_dir=db_dir) is None
    assert verified_image_file_mime(fake_header) == ""

    resolved = resolve_verified_chat_image_upload_path(valid, db_dir=db_dir)
    assert resolved is not None
    assert resolved[0] == valid.resolve(strict=False)
    assert resolved[1] == "image/png"


def test_resolve_verified_chat_image_upload_path_accepts_leaf_name_only(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    image_dir = db_dir / "_chat_uploads" / "images"
    image_dir.mkdir(parents=True)
    valid = image_dir / "valid.png"
    valid.write_bytes(TINY_PNG_BYTES)
    nested = image_dir / "nested"
    nested.mkdir()
    (nested / "valid.png").write_bytes(TINY_PNG_BYTES)

    resolved = resolve_verified_chat_image_upload_path("valid.png", db_dir=db_dir)

    assert resolved is not None
    assert resolved[0] == valid.resolve(strict=False)
    assert resolve_verified_chat_image_upload_path("nested/valid.png", db_dir=db_dir) is None
    assert resolve_verified_chat_image_upload_path("../valid.png", db_dir=db_dir) is None


def test_unique_resolved_roots_deduplicates_equivalent_paths(tmp_path: Path) -> None:
    root = tmp_path / "library"
    root.mkdir()

    roots = unique_resolved_roots([root, str(root), root.parent / "library" / ".." / "library"])

    assert roots == [root.resolve(strict=False)]


def test_path_is_within_roots_rejects_sibling_prefix(tmp_path: Path) -> None:
    root = tmp_path / "library"
    sibling = tmp_path / "library-extra"
    root.mkdir()
    sibling.mkdir()
    candidate = sibling / "paper.pdf"
    candidate.write_bytes(b"%PDF-1.4")

    assert path_is_within_roots(candidate, [root]) is False


def test_resolve_existing_file_under_roots_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "uploads"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    target = outside / "secret.png"
    target.write_bytes(TINY_PNG_BYTES)
    link = root / "linked.png"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable in this environment: {exc}")

    assert resolve_existing_file_under_roots(link, [root]) is None


def test_resolve_verified_pdf_file_under_roots_requires_pdf_content_and_root(tmp_path: Path) -> None:
    root = tmp_path / "pdfs"
    root.mkdir()
    valid = root / "paper.pdf"
    valid.write_bytes(b"%PDF-1.4\nbody")
    leading_whitespace = root / "leading.pdf"
    leading_whitespace.write_bytes(b"\xef\xbb\xbf\n  %PDF-1.7\nbody")
    fake = root / "fake.pdf"
    fake.write_bytes(b"not a pdf")
    fake_embedded_header = root / "fake-embedded.pdf"
    fake_embedded_header.write_bytes(b"not a pdf even if it mentions %PDF-1.4 later")
    wrong_suffix = root / "paper.txt"
    wrong_suffix.write_bytes(b"%PDF-1.4\nbody")
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\nbody")

    assert is_probably_pdf_bytes(b"prefix %PDF-1.4") is False
    assert is_probably_pdf_bytes(b"\n  %PDF-1.4") is True
    assert resolve_verified_pdf_file_under_roots(valid, [root]) == valid.resolve(strict=False)
    assert resolve_verified_pdf_file_under_roots(leading_whitespace, [root]) == leading_whitespace.resolve(strict=False)
    assert resolve_verified_pdf_file_under_roots(fake, [root]) is None
    assert resolve_verified_pdf_file_under_roots(fake_embedded_header, [root]) is None
    assert resolve_verified_pdf_file_under_roots(wrong_suffix, [root]) is None
    assert resolve_verified_pdf_file_under_roots(outside, [root]) is None
