from __future__ import annotations

import time

from ui.refs_renderer import _core_parse_rows


def test_core_parse_rows_reads_nested_result_cells() -> None:
    html = """
    <table>
      <tr onclick="navigate('/conf-ranks/123')">
        <td><strong>International Conference on Computer Vision</strong></td>
        <td><span>ICCV</span></td>
        <td>CORE 2023</td>
        <td><b>A*</b></td>
      </tr>
      <tr><td>Not a result</td><td>NOPE</td><td>Other</td><td>A</td></tr>
    </table>
    """

    assert _core_parse_rows(html) == [
        {
            "title": "International Conference on Computer Vision",
            "acronym": "ICCV",
            "source": "CORE 2023",
            "rank": "A*",
        }
    ]


def test_core_parse_rows_stays_bounded_on_large_malformed_response() -> None:
    malformed = "<html>" + ("<tr onclick=\"navigate('/x')\"><td>noise" * 80_000)
    started = time.perf_counter()

    assert _core_parse_rows(malformed) == []
    assert time.perf_counter() - started < 2.0
