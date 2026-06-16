from pathlib import Path

import pytest
from fastapi import HTTPException

from api.library_path_utils import (
    path_is_within,
    resolve_library_pdf_name_arg,
    resolve_library_pdf_path_arg,
)


def test_path_is_within_allows_children_and_rejects_siblings(tmp_path: Path):
    root = tmp_path / "pdfs"
    sibling = tmp_path / "outside"
    root.mkdir()
    sibling.mkdir()

    assert path_is_within(root / "paper.pdf", [root]) is True
    assert path_is_within(sibling / "paper.pdf", [root]) is False


def test_resolve_library_pdf_path_arg_requires_pdf_under_root(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    resolved = resolve_library_pdf_path_arg("paper.pdf", pdf_dir=pdf_dir)

    assert resolved == (pdf_dir / "paper.pdf").resolve(strict=False)

    with pytest.raises(HTTPException) as exc_info:
        resolve_library_pdf_path_arg("../outside/paper.pdf", pdf_dir=pdf_dir)

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("bad_name", ["../paper.pdf", "nested/paper.pdf", r"nested\paper.pdf", "paper.md", ""])
def test_resolve_library_pdf_name_arg_rejects_non_leaf_or_non_pdf_names(tmp_path: Path, bad_name: str):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    with pytest.raises(HTTPException):
        resolve_library_pdf_name_arg(bad_name, pdf_dir=pdf_dir)


def test_resolve_library_pdf_name_arg_can_require_existing_file(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "paper.pdf"

    with pytest.raises(HTTPException) as exc_info:
        resolve_library_pdf_name_arg("paper.pdf", pdf_dir=pdf_dir, require_exists=True, is_file=lambda _path: False)

    assert exc_info.value.status_code == 404

    resolved = resolve_library_pdf_name_arg("paper.pdf", pdf_dir=pdf_dir, require_exists=True, is_file=lambda path: path == pdf_path)
    assert resolved == pdf_path.resolve(strict=False)
