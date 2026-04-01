import time
from pathlib import Path

from kb.converter.config import ConvertConfig
from kb.converter.pipeline import PDFConverter
import kb.converter.pipeline as pipeline_module


def _make_converter(tmp_path):
    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    return PDFConverter(cfg)


class _DummyPage:
    def __init__(self, idx: int):
        self.idx = idx


class _DummyDoc:
    def __init__(self, total_pages: int):
        self._pages = [_DummyPage(i) for i in range(total_pages)]

    def __len__(self):
        return len(self._pages)

    def load_page(self, idx: int):
        return self._pages[idx]


class _OpenedDoc:
    def __init__(self, total_pages: int, owner=None):
        self._doc = _DummyDoc(total_pages)
        self._owner = owner

    def load_page(self, idx: int):
        return self._doc.load_page(idx)

    def close(self):
        if self._owner is not None:
            self._owner.close_calls += 1
        return None


class _FakeFitz:
    def __init__(self, total_pages: int):
        self._total_pages = total_pages
        self.open_calls = 0
        self.close_calls = 0

    def open(self, pdf_path):
        self.open_calls += 1
        return _OpenedDoc(self._total_pages, owner=self)


def test_process_batch_no_llm_parallel_matches_sequential_outputs(tmp_path, monkeypatch):
    total_pages = 4
    doc = _DummyDoc(total_pages)
    converter = _make_converter(tmp_path)

    monkeypatch.setattr(converter, "_get_speed_mode_config", lambda speed_mode, total: {"max_parallel_pages": 3})
    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: f"MD_{page_index}",
    )

    monkeypatch.setenv("KB_PDF_NO_LLM_PAGE_WORKERS", "1")
    seq = converter._process_batch_no_llm(doc, pdf_path=Path("dummy.pdf"), assets_dir=tmp_path)

    monkeypatch.setenv("KB_PDF_NO_LLM_PAGE_WORKERS", "3")
    monkeypatch.setattr(pipeline_module, "fitz", _FakeFitz(total_pages))
    monkeypatch.setattr(
        pipeline_module,
        "prepare_page_render_input",
        lambda converter, page, page_index, pdf_path, assets_dir: {
            "blocks": [f"block_{page_index}"],
            "is_references_page": False,
            "prepare_elapsed": 0.01,
            "page_index": page_index,
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "render_prepared_page",
        lambda converter, *, prepared, page, page_index, assets_dir: f"MD_{prepared['page_index']}",
    )

    par = converter._process_batch_no_llm(doc, pdf_path=Path("dummy.pdf"), assets_dir=tmp_path)

    assert par == seq


def test_process_batch_no_llm_parallel_renders_in_page_order(tmp_path, monkeypatch):
    total_pages = 3
    doc = _DummyDoc(total_pages)
    converter = _make_converter(tmp_path)
    render_order = []

    monkeypatch.setenv("KB_PDF_NO_LLM_PAGE_WORKERS", "3")
    monkeypatch.setattr(converter, "_get_speed_mode_config", lambda speed_mode, total: {"max_parallel_pages": 3})
    monkeypatch.setattr(pipeline_module, "fitz", _FakeFitz(total_pages))

    def _prepare(converter, page, page_index, pdf_path, assets_dir):
        time.sleep(0.03 * (total_pages - page_index))
        return {
            "blocks": [f"block_{page_index}"],
            "is_references_page": False,
            "prepare_elapsed": 0.01,
            "page_index": page_index,
        }

    def _render(converter, *, prepared, page, page_index, assets_dir):
        render_order.append(page_index)
        return f"MD_{page_index}"

    monkeypatch.setattr(pipeline_module, "prepare_page_render_input", _prepare)
    monkeypatch.setattr(pipeline_module, "render_prepared_page", _render)

    out = converter._process_batch_no_llm(doc, pdf_path=Path("dummy.pdf"), assets_dir=tmp_path)

    assert out == ["MD_0", "MD_1", "MD_2"]
    assert render_order == [0, 1, 2]


def test_process_batch_no_llm_parallel_reuses_worker_local_documents(tmp_path, monkeypatch):
    total_pages = 4
    doc = _DummyDoc(total_pages)
    converter = _make_converter(tmp_path)
    fake_fitz = _FakeFitz(total_pages)

    monkeypatch.setenv("KB_PDF_NO_LLM_PAGE_WORKERS", "2")
    monkeypatch.setattr(converter, "_get_speed_mode_config", lambda speed_mode, total: {"max_parallel_pages": 2})
    monkeypatch.setattr(pipeline_module, "fitz", fake_fitz)

    def _prepare(converter, page, page_index, pdf_path, assets_dir):
        time.sleep(0.03)
        return {
            "blocks": [f"block_{page_index}"],
            "is_references_page": False,
            "prepare_elapsed": 0.01,
            "page_index": page_index,
        }

    monkeypatch.setattr(pipeline_module, "prepare_page_render_input", _prepare)
    monkeypatch.setattr(
        pipeline_module,
        "render_prepared_page",
        lambda converter, *, prepared, page, page_index, assets_dir: f"MD_{prepared['page_index']}",
    )

    out = converter._process_batch_no_llm(doc, pdf_path=Path("dummy.pdf"), assets_dir=tmp_path)

    assert out == ["MD_0", "MD_1", "MD_2", "MD_3"]
    assert fake_fitz.open_calls < total_pages
    assert fake_fitz.open_calls <= 2
    assert fake_fitz.close_calls == fake_fitz.open_calls
