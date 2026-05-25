from __future__ import annotations

import api.routers.references as references_router


class _ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, **_kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self) -> None:
        self.target(*self.args, **self.kwargs)


def _clear_card_polish_state() -> None:
    references_router._CITATION_CARD_POLISH_CACHE.clear()
    references_router._CITATION_CARD_POLISH_WARMING.clear()


def test_citation_card_polish_route_returns_pending_then_cached(monkeypatch) -> None:
    _clear_card_polish_state()
    monkeypatch.setattr(references_router.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(references_router, "citation_card_polish_enabled", lambda: True)
    monkeypatch.setattr(
        references_router,
        "polish_citation_card_detail",
        lambda _detail: {
            "citation_card_polish_status": "full",
            "citation_card_polish_source": "llm",
            "citation_card_polish_checked": True,
            "card_takeaway": "Polished card takeaway",
        },
    )
    body = references_router.CitationCardPolishBody(
        meta={
            "sourceName": "Fixture.pdf",
            "headingPath": "Abstract",
            "answerClaim": "The answer uses the paper as evidence.",
            "evidenceQuote": "The paper states the relevant mechanism in the abstract.",
        }
    )

    first = references_router.polish_citation_card(body)
    second = references_router.polish_citation_card(body)

    assert first["citation_card_polish_status"] == "pending"
    assert first["citation_card_polish_started"] is True
    assert second["citation_card_polish_status"] == "full"
    assert second["citation_card_polish_cached"] is True
    assert second["card_takeaway"] == "Polished card takeaway"
    assert second["citation_card_polish_key"]


def test_citation_card_polish_route_can_wait_for_fresh_polish(monkeypatch) -> None:
    _clear_card_polish_state()
    monkeypatch.setattr(references_router.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(references_router, "citation_card_polish_enabled", lambda: True)
    monkeypatch.setattr(
        references_router,
        "polish_citation_card_detail",
        lambda _detail: {
            "citation_card_polish_status": "full",
            "citation_card_polish_source": "llm",
            "citation_card_polish_checked": True,
            "card_takeaway": "Polished immediately after waiting",
        },
    )
    body = references_router.CitationCardPolishBody(
        wait_s=1.0,
        meta={
            "sourceName": "Fixture.pdf",
            "headingPath": "Abstract",
            "answerClaim": "The answer uses the paper as evidence.",
            "evidenceQuote": "The paper states the relevant mechanism in the abstract.",
        },
    )

    out = references_router.polish_citation_card(body)

    assert out["citation_card_polish_status"] == "full"
    assert out["citation_card_polish_cached"] is True
    assert out["citation_card_polish_waited"] is True
    assert out["card_takeaway"] == "Polished immediately after waiting"


def test_citation_card_polish_route_respects_disabled_flag(monkeypatch) -> None:
    _clear_card_polish_state()
    monkeypatch.setattr(references_router, "citation_card_polish_enabled", lambda: False)
    monkeypatch.setattr(
        references_router,
        "_schedule_citation_card_polish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("disabled route should not warm")),
    )
    body = references_router.CitationCardPolishBody(
        meta={
            "source_name": "Fixture.pdf",
            "heading_path": "Abstract",
            "answer_claim": "The answer uses the paper as evidence.",
            "evidence_quote": "The paper states the relevant mechanism in the abstract.",
        }
    )

    out = references_router.polish_citation_card(body)

    assert out["citation_card_polish_status"] == "disabled"
    assert out["citation_card_polish_checked"] is True
