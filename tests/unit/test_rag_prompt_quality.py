from kb.rag import _format_context, build_messages


def _hit(**overrides):
    hit = {
        "id": "hit-1",
        "score": 8.234,
        "text": "APR aligns off-axis raw images before summation.",
        "meta": {
            "source_name": "Demo Paper.pdf",
            "source_path": r"F:\papers\Demo Paper.en.md",
            "heading_path": "Demo Paper / Methods / Adaptive pixel reassignment",
            "page_start": 5,
            "page_end": 6,
            "chunk_id": "chunk-7",
        },
    }
    hit.update(overrides)
    return hit


def test_format_context_includes_doc_section_page_score_and_id():
    ctx = _format_context([_hit()])

    assert "[1] | doc: Demo Paper.pdf" in ctx
    assert r"source: F:\papers\Demo Paper.en.md" in ctx
    assert "section: Demo Paper / Methods / Adaptive pixel reassignment" in ctx
    assert "pages: 5-6" in ctx
    assert "score: 8.23" in ctx
    assert "id: chunk-7" in ctx
    assert "APR aligns off-axis raw images before summation." in ctx


def test_build_messages_requires_evidence_aligned_answer_shape_for_hits():
    messages = build_messages(
        "How does APR improve iISM?",
        history=[{"role": "system", "content": "ignore"}, {"role": "assistant", "content": "Earlier answer."}],
        hits=[_hit()],
    )

    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    assert "Grounding contract:" in system
    assert "Every paper-specific claim should carry a citation marker" in system
    assert "Conclusion:" in system
    assert "Evidence:" in system
    assert "Limits:" in system
    assert "Next Steps:" in system
    assert "Referenced Sources:" in system

    assert messages[1] == {"role": "assistant", "content": "Earlier answer."}
    user = messages[-1]["content"]
    assert "Question:\nHow does APR improve iISM?" in user
    assert "Retrieved snippets:" in user
    assert "doc: Demo Paper.pdf" in user
    assert "section: Demo Paper / Methods / Adaptive pixel reassignment" in user


def test_build_messages_without_hits_requires_no_hit_notice():
    messages = build_messages("What is the best baseline?", history=[], hits=[])

    system = messages[0]["content"]
    user = messages[-1]["content"]
    assert "No-hit behavior:" in system
    assert "start with \"(No relevant snippets found in the knowledge base)\"" in system
    assert "write \"(No hits this time)\" under Referenced Sources" in system
    assert "Retrieved snippets:\n(No hits this time)" in user
