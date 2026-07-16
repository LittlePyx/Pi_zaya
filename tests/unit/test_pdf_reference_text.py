from __future__ import annotations

from types import SimpleNamespace

from kb.converter.pdf_reference_text import (
    consecutive_reference_chain_positions,
    is_reference_running_line,
    merge_standalone_reference_continuations,
    reference_ordered_page_text,
    trim_reference_publisher_tail,
)


def _line(text: str, x0: float, y0: float, x1: float) -> dict:
    return {"bbox": (x0, y0, x1, y0 + 8.0), "spans": [{"text": text}]}


def test_reference_number_chain_ignores_volume_number_noise() -> None:
    numbers = [3, 1, 2, 13, 3, 4, 5, 104, 6]

    positions = consecutive_reference_chain_positions(numbers)

    assert [numbers[idx] for idx in positions] == [1, 2, 3, 4, 5, 6]


def test_bracketed_reference_page_joins_standalone_page_numbers() -> None:
    text = "\n".join(
        [
            "[1] First source. Journal 2020, 10, 100.",
            "[2] Second source. Nature 2019, 13,",
            "13.",
            "[3] Third source. Optics 2021, 4, 44.",
        ]
    )

    merged = merge_standalone_reference_continuations(text)

    assert "Nature 2019, 13, 13." in merged
    assert "\n13.\n" not in f"\n{merged}\n"


def test_bracketed_reference_page_keeps_a_real_standalone_next_entry() -> None:
    text = "\n".join(
        [
            "[10] First source. Journal 2020, 10, 100.",
            "[11] Second source. Journal 2021, 11, 110.",
            "[12] Third source. Journal 2022, 3, 3.",
            "13.",
            "Doe, D. Fourth source. Journal of Tests 2023, 4, 44.",
        ]
    )

    merged = merge_standalone_reference_continuations(text)

    assert "\n13.\nDoe, D. Fourth source" in f"\n{merged}\n"


def test_interleaved_competing_reference_chains_are_ambiguous() -> None:
    assert consecutive_reference_chain_positions([1, 20, 2, 21, 3, 22]) == []


def test_transition_page_uses_column_order_despite_spanning_table_cells() -> None:
    lines = [
        *[_line(f"Left prose anchor {idx}", 44.0, 60.0 + idx * 14.0, 291.0) for idx in range(6)],
        _line("Information", 253.0, 82.0, 335.0),
        *[_line(f"Right reference prose {idx}", 304.0, 50.0 + idx * 14.0, 552.0) for idx in range(6)],
    ]

    class _Page:
        rect = SimpleNamespace(width=595.0)

        def get_text(self, mode: str):
            assert mode == "dict"
            return {"blocks": [{"lines": lines}]}

    ordered = reference_ordered_page_text(_Page(), fallback_text="fallback")

    assert ordered.index("Left prose anchor 5") < ordered.index("Right reference prose 0")


def test_publisher_running_lines_are_recognized() -> None:
    assert is_reference_running_line("Article")
    assert is_reference_running_line(
        "NATURE COMMUNICATIONS | (2021) 12:4712 | https://doi.org/example"
    )
    assert is_reference_running_line("Laser Photonics Rev. 2025, 19, 2401397")
    assert is_reference_running_line("2401397 (20 of 21)")
    assert not is_reference_running_line("Article title with meaningful content")
    assert not is_reference_running_line("[20] A real reference. Nature 2021, 1, 20.")


def test_publisher_note_does_not_extend_the_final_reference() -> None:
    text = "\n".join(
        [
            "57. A real source. Nature 2025, 1, 10.",
            "58. Final source. Opt. Lett. 46, 2884–2887 (2021).",
            "Publisher’s note Springer Nature remains neutral with regard to",
            "jurisdictional claims in published maps and institutional affiliations.",
            "Open Access This article is licensed under a Creative Commons licence.",
        ]
    )

    trimmed = trim_reference_publisher_tail(text)

    assert trimmed.endswith("58. Final source. Opt. Lett. 46, 2884–2887 (2021).")
    assert "Publisher" not in trimmed
    assert "Open Access" not in trimmed
