from __future__ import annotations

from kb.retriever import BM25Retriever


def test_retriever_excludes_chunks_marked_unreliable_for_evidence() -> None:
    retriever = BM25Retriever(
        [
            {
                "id": "damaged",
                "text": "photon imaging reconstruction advantage advantage advantage",
                "meta": {"evidence_ready": False, "page_start": 10},
            },
            {
                "id": "clean",
                "text": "photon imaging reconstruction provides robust evidence",
                "meta": {"evidence_ready": True, "page_start": 11},
            },
            {"id": "other-1", "text": "spectral calibration workflow", "meta": {}},
            {"id": "other-2", "text": "detector noise measurement", "meta": {}},
            {"id": "other-3", "text": "optical alignment protocol", "meta": {}},
        ]
    )

    hits = retriever.search("photon imaging reconstruction advantage", top_k=5)

    assert hits[0]["id"] == "clean"
    assert all(hit["id"] != "damaged" for hit in hits)
