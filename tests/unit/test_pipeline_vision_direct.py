import time
from pathlib import Path

from kb.converter.config import ConvertConfig
from kb.converter.pipeline import PDFConverter
import kb.converter.pipeline_vision_direct as vision_direct_module


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

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

    def Matrix(self, x, y):
        return (x, y)

    def open(self, pdf_path):
        self.open_calls += 1
        return _OpenedDoc(self._total_pages, owner=self)


def test_process_batch_vision_direct_parallel_matches_sequential_outputs(tmp_path, monkeypatch):
    total_pages = 4
    doc = _DummyDoc(total_pages)
    converter = _make_converter(tmp_path)

    monkeypatch.setattr(
        converter,
        "_get_speed_mode_config",
        lambda speed_mode, total: {"max_parallel_pages": 3, "dpi": 220, "compress": 3, "max_inflight": 8},
    )
    monkeypatch.setattr(converter.llm_worker, "get_llm_max_inflight", lambda: 4)
    monkeypatch.setattr(vision_direct_module, "fitz", _FakeFitz(total_pages))
    monkeypatch.setattr(
        vision_direct_module,
        "process_vision_direct_page",
        lambda converter, *, page, page_index, total_pages, pdf_path, assets_dir, speed_mode, speed_config, dpi, mat, started_at=None: f"MD_{page_index}",
    )

    monkeypatch.setenv("KB_PDF_LLM_PAGE_WORKERS", "1")
    seq = converter._process_batch_vision_direct(doc, pdf_path=Path("dummy.pdf"), assets_dir=tmp_path)

    monkeypatch.setenv("KB_PDF_LLM_PAGE_WORKERS", "3")
    par = converter._process_batch_vision_direct(doc, pdf_path=Path("dummy.pdf"), assets_dir=tmp_path)

    assert par == seq


def test_process_batch_vision_direct_parallel_preserves_page_order(tmp_path, monkeypatch):
    total_pages = 3
    doc = _DummyDoc(total_pages)
    converter = _make_converter(tmp_path)
    completion_order = []

    monkeypatch.setenv("KB_PDF_LLM_PAGE_WORKERS", "3")
    monkeypatch.setattr(
        converter,
        "_get_speed_mode_config",
        lambda speed_mode, total: {"max_parallel_pages": 3, "dpi": 220, "compress": 3, "max_inflight": 8},
    )
    monkeypatch.setattr(converter.llm_worker, "get_llm_max_inflight", lambda: 4)
    monkeypatch.setattr(vision_direct_module, "fitz", _FakeFitz(total_pages))

    def _process_page(converter, *, page, page_index, total_pages, pdf_path, assets_dir, speed_mode, speed_config, dpi, mat, started_at=None):
        delays = {0: 0.15, 1: 0.05, 2: 0.0}
        time.sleep(delays[page_index])
        completion_order.append(page_index)
        return f"MD_{page_index}"

    monkeypatch.setattr(vision_direct_module, "process_vision_direct_page", _process_page)

    out = converter._process_batch_vision_direct(doc, pdf_path=Path("dummy.pdf"), assets_dir=tmp_path)

    assert out == ["MD_0", "MD_1", "MD_2"]
    assert completion_order == [2, 1, 0]


def test_process_batch_vision_direct_parallel_reuses_worker_local_documents(tmp_path, monkeypatch):
    total_pages = 4
    doc = _DummyDoc(total_pages)
    converter = _make_converter(tmp_path)
    fake_fitz = _FakeFitz(total_pages)

    monkeypatch.setenv("KB_PDF_LLM_PAGE_WORKERS", "2")
    monkeypatch.setattr(
        converter,
        "_get_speed_mode_config",
        lambda speed_mode, total: {"max_parallel_pages": 2, "dpi": 220, "compress": 3, "max_inflight": 8},
    )
    monkeypatch.setattr(converter.llm_worker, "get_llm_max_inflight", lambda: 4)
    monkeypatch.setattr(vision_direct_module, "fitz", fake_fitz)

    def _process_page(converter, *, page, page_index, total_pages, pdf_path, assets_dir, speed_mode, speed_config, dpi, mat, started_at=None):
        time.sleep(0.03)
        return f"MD_{page_index}"

    monkeypatch.setattr(vision_direct_module, "process_vision_direct_page", _process_page)

    out = converter._process_batch_vision_direct(doc, pdf_path=Path("dummy.pdf"), assets_dir=tmp_path)

    assert out == ["MD_0", "MD_1", "MD_2", "MD_3"]
    assert fake_fitz.open_calls < total_pages
    assert fake_fitz.open_calls <= 2
    assert fake_fitz.close_calls == fake_fitz.open_calls
