from __future__ import annotations

from pathlib import Path

from kb.retriever_cache import clear_retriever_cache, get_cached_retriever
from kb.store import load_docs_index, save_docs_index, write_doc_chunks


def _publish_doc(db_dir: Path, doc_id: str, text: str) -> None:
    write_doc_chunks(
        db_dir,
        doc_id,
        [{"text": text, "meta": {"source_path": f"{doc_id}.pdf"}}],
    )
    docs = load_docs_index(db_dir)
    docs[doc_id] = {"path": f"{doc_id}.md", "index_status": "ready", "num_chunks": 1}
    save_docs_index(db_dir, docs)


def test_retriever_cache_reuses_index_and_isolates_hit_metadata(tmp_path: Path) -> None:
    clear_retriever_cache()
    _publish_doc(tmp_path, "paper-a", "ADMM reconstruction improves image quality")
    _publish_doc(tmp_path, "filler-a", "quantum optics experiment")
    _publish_doc(tmp_path, "filler-b", "biological microscopy analysis")

    first, first_count, first_hit = get_cached_retriever(tmp_path)
    second, second_count, second_hit = get_cached_retriever(tmp_path)

    assert first is second
    assert first_count == second_count == 3
    assert first_hit is False
    assert second_hit is True
    hit = first.search("ADMM reconstruction", top_k=1)[0]
    hit["meta"]["conversation_only"] = True
    assert "conversation_only" not in second.search("ADMM reconstruction", top_k=1)[0]["meta"]


def test_retriever_cache_rebuilds_after_docs_index_commit(tmp_path: Path) -> None:
    clear_retriever_cache()
    _publish_doc(tmp_path, "paper-a", "first reconstruction method")
    _publish_doc(tmp_path, "filler-a", "quantum optics experiment")
    _publish_doc(tmp_path, "filler-b", "biological microscopy analysis")
    first, _, _ = get_cached_retriever(tmp_path)

    _publish_doc(tmp_path, "paper-b", "second reconstruction method")
    second, count, cache_hit = get_cached_retriever(tmp_path)

    assert second is not first
    assert count == 4
    assert cache_hit is False
    assert second.search("second reconstruction", top_k=1)[0]["meta"]["source_path"] == "paper-b.pdf"
