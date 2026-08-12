from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_FIXTURE = Path("docs/research_qa_unseen_corpus_v1.json")


def load_corpus_manifest(path: Path | str = DEFAULT_FIXTURE) -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_corpus = dict(data.get("benchmark") or {}).get("corpus") or []
    corpus: list[dict[str, str]] = []
    for raw in raw_corpus:
        item = dict(raw) if isinstance(raw, dict) else {}
        doc_id = str(item.get("id") or "").strip()
        arxiv_id = str(item.get("arxiv") or "").strip()
        sha256 = str(item.get("sha256") or "").strip().lower()
        if not doc_id or not arxiv_id or len(sha256) != 64:
            raise ValueError(f"invalid unseen-corpus manifest item: {raw!r}")
        corpus.append({"id": doc_id, "arxiv": arxiv_id, "sha256": sha256})
    if not corpus:
        raise ValueError("unseen-corpus manifest is empty")
    if len({item["id"] for item in corpus}) != len(corpus):
        raise ValueError("unseen-corpus manifest contains duplicate ids")
    return corpus


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pdf(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        signature = handle.read(5)
    if signature != b"%PDF-":
        raise ValueError(f"not a PDF: {path}")
    actual = sha256_file(path)
    if actual.casefold() != str(expected_sha256).casefold():
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual}")


def download_pdf(arxiv_id: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        f"https://arxiv.org/pdf/{arxiv_id}",
        headers={"User-Agent": "Pi-zaya-research-qa-eval/1.0"},
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.stem}.",
            suffix=".pdf.tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            with urlopen(request, timeout=120) as response:  # noqa: S310 - fixed arxiv host
                while block := response.read(1024 * 1024):
                    output.write(block)
        verify_pdf(temp_path, expected_sha256)
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def prepare_corpus(
    *,
    fixture_path: Path,
    out_dir: Path,
    verify_only: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in load_corpus_manifest(fixture_path):
        destination = out_dir / f"{item['id']}.pdf"
        downloaded = False
        if not destination.is_file():
            if verify_only:
                raise FileNotFoundError(destination)
            download_pdf(item["arxiv"], destination, item["sha256"])
            downloaded = True
        verify_pdf(destination, item["sha256"])
        results.append(
            {
                "id": item["id"],
                "path": str(destination.resolve()),
                "sha256": item["sha256"],
                "downloaded": downloaded,
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and SHA-256 verify the pre-registered unseen-paper QA corpus."
    )
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--out-dir", required=True, help="External directory for PDF binaries.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing corpus without downloading missing PDFs.",
    )
    args = parser.parse_args(argv)

    try:
        results = prepare_corpus(
            fixture_path=Path(args.fixture),
            out_dir=Path(args.out_dir),
            verify_only=bool(args.verify_only),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    for result in results:
        action = "downloaded" if result["downloaded"] else "verified"
        print(f"[OK] {result['id']}: {action} {result['sha256']}")
    print(f"[OK] corpus: {len(results)} PDFs in {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
